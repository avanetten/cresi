#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code from the osmnx v0.9 package.
"""

import time
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
# import os
# import matplotlib.cm as cm
# from descartes import PolygonPatch


###############################################################################
# https://github.com/gboeing/osmnx/blob/master/osmnx/settings.py
# default CRS to set when creating graphs
default_crs = {'init': 'epsg:4326'}


###############################################################################
def project_gdf(gdf, to_crs=None, to_latlong=False, verbose=False):
    """
    https://github.com/gboeing/osmnx/blob/v0.9/osmnx/projection.py#L58

    Project a GeoDataFrame to the UTM zone appropriate for its geometries'
    centroid.
    The simple calculation in this function works well for most latitudes, but
    won't work for some far northern locations like Svalbard and parts of far
    northern Norway.
    Parameters
    ----------
    gdf : GeoDataFrame
        the gdf to be projected
    to_crs : dict
        if not None, just project to this CRS instead of to UTM
    to_latlong : bool
        if True, projects to latlong instead of to UTM
    Returns
    -------
    GeoDataFrame
    """
    assert len(gdf) > 0, 'You cannot project an empty GeoDataFrame.'
    start_time = time.time()

    # if gdf has no gdf_name attribute, create one now
    if not hasattr(gdf, 'gdf_name'):
        gdf.gdf_name = 'unnamed'

    # if to_crs was passed-in, use this value to project the gdf
    if to_crs is not None:
        projected_gdf = gdf.to_crs(to_crs)

    # if to_crs was not passed-in, calculate the centroid of the geometry to
    # determine UTM zone
    else:
        if to_latlong:
            # if to_latlong is True, project the gdf to latlong
            latlong_crs = default_crs
            projected_gdf = gdf.to_crs(latlong_crs)
            if verbose:
                print('Projected the GeoDataFrame "{}" to default_crs in {:,.2f} seconds'.format(gdf.gdf_name, time.time()-start_time))
        else:
            # else, project the gdf to UTM
            # if GeoDataFrame is already in UTM, just return it
            if (gdf.crs is not None) and ('proj' in gdf.crs) and (gdf.crs['proj'] == 'utm'):
                return gdf

            # calculate the centroid of the union of all the geometries in the
            # GeoDataFrame
            avg_longitude = gdf['geometry'].unary_union.centroid.x

            # calculate the UTM zone from this avg longitude and define the UTM
            # CRS to project
            utm_zone = int(math.floor((avg_longitude + 180) / 6.) + 1)
            utm_crs = {'datum': 'WGS84',
                       'ellps': 'WGS84',
                       'proj' : 'utm',
                       'zone' : utm_zone,
                       'units': 'm'}

            # project the GeoDataFrame to the UTM CRS
            projected_gdf = gdf.to_crs(utm_crs)
            if verbose:
                print('Projected the GeoDataFrame "{}" to UTM-{} in {:,.2f} seconds'.format(gdf.gdf_name, utm_zone, time.time()-start_time))

    projected_gdf.gdf_name = gdf.gdf_name
    return projected_gdf


###############################################################################
def project_graph(G, to_crs=None, verbose=False):
    """
    https://github.com/gboeing/osmnx/blob/v0.9/osmnx/projection.py#L126
    Project a graph from lat-long to the UTM zone appropriate for its geographic
    location.
    Parameters
    ----------
    G : networkx multidigraph
        the networkx graph to be projected
    to_crs : dict
        if not None, just project to this CRS instead of to UTM
    Returns
    -------
    networkx multidigraph
    """

    G_proj = G.copy()
    start_time = time.time()

    # create a GeoDataFrame of the nodes, name it, convert osmid to str
    nodes, data = zip(*G_proj.nodes(data=True))
    gdf_nodes = gpd.GeoDataFrame(list(data), index=nodes)
    gdf_nodes.crs = G_proj.graph['crs']
    gdf_nodes.gdf_name = '{}_nodes'.format(G_proj.name)

    # create new lat/lon columns just to save that data for later, and create a
    # geometry column from x/y
    gdf_nodes['lon'] = gdf_nodes['x']
    gdf_nodes['lat'] = gdf_nodes['y']
    gdf_nodes['geometry'] = gdf_nodes.apply(lambda row: Point(row['x'], row['y']), axis=1)
    if verbose:
        print('Created a GeoDataFrame from graph in {:,.2f} seconds'.format(time.time()-start_time))

    # project the nodes GeoDataFrame to UTM
    gdf_nodes_utm = project_gdf(gdf_nodes, to_crs=to_crs)

    # extract data for all edges that have geometry attribute
    edges_with_geom = []
    for u, v, key, data in G_proj.edges(keys=True, data=True):
        if 'geometry' in data:
            edges_with_geom.append({'u':u, 'v':v, 'key':key, 'geometry':data['geometry']})

    # create an edges GeoDataFrame and project to UTM, if there were any edges
    # with a geometry attribute. geom attr only exists if graph has been
    # simplified, otherwise you don't have to project anything for the edges
    # because the nodes still contain all spatial data
    if len(edges_with_geom) > 0:
        gdf_edges = gpd.GeoDataFrame(edges_with_geom)
        gdf_edges.crs = G_proj.graph['crs']
        gdf_edges.gdf_name = '{}_edges'.format(G_proj.name)
        gdf_edges_utm = project_gdf(gdf_edges, to_crs=to_crs)

    # extract projected x and y values from the nodes' geometry column
    start_time = time.time()
    gdf_nodes_utm['x'] = gdf_nodes_utm['geometry'].map(lambda point: point.x)
    gdf_nodes_utm['y'] = gdf_nodes_utm['geometry'].map(lambda point: point.y)
    gdf_nodes_utm = gdf_nodes_utm.drop('geometry', axis=1)
    if verbose:
        print('Extracted projected node geometries from GeoDataFrame in {:,.2f} seconds'.format(time.time()-start_time))

    # clear the graph to make it a blank slate for the projected data
    start_time = time.time()
    edges = list(G_proj.edges(keys=True, data=True))
    graph_name = G_proj.graph['name']
    G_proj.clear()

    # add the projected nodes and all their attributes to the graph
    G_proj.add_nodes_from(gdf_nodes_utm.index)
    attributes = gdf_nodes_utm.to_dict()
    for label in gdf_nodes_utm.columns:
        nx.set_node_attributes(G_proj, name=label, values=attributes[label])

    # add the edges and all their attributes (including reconstructed geometry,
    # when it exists) to the graph
    for u, v, key, attributes in edges:
        if 'geometry' in attributes:
            row = gdf_edges_utm[(gdf_edges_utm['u']==u) & (gdf_edges_utm['v']==v) & (gdf_edges_utm['key']==key)]
            attributes['geometry'] = row['geometry'].iloc[0]

        # attributes dict contains key, so we don't need to explicitly pass it here
        G_proj.add_edge(u, v, **attributes)

    # set the graph's CRS attribute to the new, projected CRS and return the
    # projected graph
    G_proj.graph['crs'] = gdf_nodes_utm.crs
    G_proj.graph['name'] = '{}_UTM'.format(graph_name)
    if 'streets_per_node' in G.graph:
        G_proj.graph['streets_per_node'] = G.graph['streets_per_node']
    if verbose:
        print('Rebuilt projected graph in {:,.2f} seconds'.format(time.time()-start_time))
    return G_proj


# https://github.com/gboeing/osmnx/blob/master/osmnx/save_load.py
def graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True,
                  fill_edge_geometry=True, verbose=False):
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

        nodes, data = zip(*G.nodes(data=True))
        gdf_nodes = gpd.GeoDataFrame(list(data), index=nodes)
        if node_geometry:
            gdf_nodes['geometry'] = gdf_nodes.apply(lambda row: Point(row['x'], row['y']), axis=1)
        gdf_nodes.crs = G.graph['crs']
        gdf_nodes.gdf_name = '{}_nodes'.format(G.graph['name'])

        to_return.append(gdf_nodes)
        if verbose:
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
            if 'geometry' not in data:
                if fill_edge_geometry:
                    point_u = Point((G.nodes[u]['x'], G.nodes[u]['y']))
                    point_v = Point((G.nodes[v]['x'], G.nodes[v]['y']))
                    edge_details['geometry'] = LineString([point_u, point_v])
                else:
                    edge_details['geometry'] = np.nan

            edges.append(edge_details)

        # create a GeoDataFrame from the list of edges and set the CRS
        gdf_edges = gpd.GeoDataFrame(edges)
        gdf_edges.crs = G.graph['crs']
        gdf_edges.gdf_name = '{}_edges'.format(G.graph['name'])

        to_return.append(gdf_edges)
        if verbose:
            print('Created GeoDataFrame "{}" from graph in {:,.2f} seconds'.format(gdf_edges.gdf_name, time.time()-start_time))

    if len(to_return) > 1:
        return tuple(to_return)
    else:
        return to_return[0]


# https://github.com/gboeing/osmnx/blob/master/osmnx/plot.py
def plot_graph(G, bbox=None, fig_height=6, fig_width=None, margin=0.02,
               axis_off=True, equal_aspect=False, bgcolor='w', show=True,
               save=False, close=True, file_format='png', filename='',
               dpi=300, annotate=False, node_color='#66ccff', node_size=15,
               node_alpha=1, node_edgecolor='none', node_zorder=1,
               edge_color='#999999', edge_linewidth=1, edge_alpha=1,
               use_geom=True, verbose=False):
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

    # print('Begin plotting the graph...')
    node_Xs = [float(x) for _, x in G.nodes(data='x')]
    node_Ys = [float(y) for _, y in G.nodes(data='y')]

    # get north, south, east, west values either from bbox parameter or from the
    # spatial extent of the edges' geometries
    if bbox is None:
        edges = graph_to_gdfs(G, nodes=False, fill_edge_geometry=True)
        west, south, east, north = edges.total_bounds
    else:
        north, south, east, west = bbox

    # if caller did not pass in a fig_width, calculate it proportionately from
    # the fig_height and bounding box aspect ratio
    bbox_aspect_ratio = (north-south)/(east-west)
    if fig_width is None:
        fig_width = fig_height / bbox_aspect_ratio

    # create the figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
    ax.set_facecolor(bgcolor)

    # draw the edges as lines from node to node
    start_time = time.time()
    lines = []
    for u, v, data in G.edges(keys=False, data=True):
        if 'geometry' in data and use_geom:
            # if it has a geometry attribute (a list of line segments), add them
            # to the list of lines to plot
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=edge_color, linewidths=edge_linewidth, alpha=edge_alpha, zorder=2)
    ax.add_collection(lc)
    if verbose:
        print('Drew the graph edges in {:,.2f} seconds'.format(time.time()-start_time))

    # scatter plot the nodes
    ax.scatter(node_Xs, node_Ys, s=node_size, c=node_color, alpha=node_alpha, edgecolor=node_edgecolor, zorder=node_zorder)

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
        if G.graph['crs'] == default_crs:
            coslat = np.cos((min(node_Ys) + max(node_Ys)) / 2. / 180. * np.pi)
            ax.set_aspect(1. / coslat)
            fig.canvas.draw()

    # annotate the axis with node IDs if annotate=True
    if annotate:
        for node, data in G.nodes(data=True):
            ax.annotate(node, xy=(data['x'], data['y']))

    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, file_format, dpi,
                            axis_off, filename=filename)
    return fig, ax


# https://github.com/gboeing/osmnx/blob/master/osmnx/plot.py
def node_list_to_coordinate_lines(G, node_list, use_geom=True):
    """
    Given a list of nodes, return a list of lines that together follow the path
    defined by the list of nodes.
    Parameters
    ----------
    G : networkx multidigraph
    route : list
        the route as a list of nodes
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    Returns
    -------
    lines : list of lines given as pairs ( (x_start, y_start), (x_stop, y_stop) )
    """
    edge_nodes = list(zip(node_list[:-1], node_list[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])

        # if it has a geometry attribute (ie, a list of line segments)
        if 'geometry' in data and use_geom:
            # add them to the list of lines to plot
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
    return lines


# https://github.com/gboeing/osmnx/blob/master/osmnx/plot.py
def plot_graph_route(G, route, bbox=None, fig_height=6, fig_width=None,
                     margin=0.02, bgcolor='w', axis_off=True, show=True,
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
    orig_dest_point_color : string
        the color of the origin and destination points if being plotted instead
        of nodes
    Returns
    -------
    fig, ax : tuple
    """

    # plot the graph but not the route
    fig, ax = plot_graph(G, bbox=bbox, fig_height=fig_height, fig_width=fig_width,
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
        origin_destination_lats = (G.nodes[origin_node]['y'], G.nodes[destination_node]['y'])
        origin_destination_lons = (G.nodes[origin_node]['x'], G.nodes[destination_node]['x'])
    else:
        # otherwise, use the passed points as origin/destination
        origin_destination_lats = (origin_point[0], destination_point[0])
        origin_destination_lons = (origin_point[1], destination_point[1])
        orig_dest_node_color = orig_dest_point_color

    # scatter the origin and destination points
    ax.scatter(origin_destination_lons, origin_destination_lats, s=orig_dest_node_size,
               c=orig_dest_node_color, alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)

    # plot the route lines
    lines = node_list_to_coordinate_lines(G, route, use_geom)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=route_color, linewidths=route_linewidth, alpha=route_alpha, zorder=3)
    ax.add_collection(lc)

    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, filename, file_format, dpi, axis_off)
    return fig, ax


# https://github.com/gboeing/osmnx/blob/master/osmnx/plot.py
def save_and_show(fig, ax, save, show, close, file_format, dpi, axis_off,
                  filename='', verbose=False):
    """
    Save a figure to disk and show it, as specified.
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
        # if not os.path.exists(settings.imgs_folder):
        #    os.makedirs(settings.imgs_folder)
        # path_filename = os.path.join(settings.imgs_folder, os.extsep.join([filename, file_format]))
        path_filename = filename

        if file_format == 'svg':
            # if the file_format is svg, prep the fig/ax a bit for saving
            ax.axis('off')
            ax.set_position([0, 0, 1, 1])
            ax.patch.set_alpha(0.)
            fig.patch.set_alpha(0.)
            if len(filename) > 0:
                fig.savefig(path_filename, bbox_inches=0, format=file_format, facecolor=fig.get_facecolor(), transparent=True)
        else:
            if axis_off:
                # if axis is turned off, constrain the saved figure's extent to
                # the interior of the axis
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            else:
                extent = 'tight'
            if len(filename) > 0:
                fig.savefig(path_filename, dpi=dpi, bbox_inches=extent, format=file_format, facecolor=fig.get_facecolor(), transparent=True)
        if verbose:
            print('Saved the figure to disk in {:,.2f} seconds'.format(time.time()-start_time))

    # show the figure if specified
    if show:
        start_time = time.time()
        plt.show()
        # print('Showed the plot in {:,.2f} seconds'.format(time.time()-start_time))
    # if show=False, close the figure if close=True to prevent display
    elif close:
        plt.close()

    return fig, ax


# https://github.com/gboeing/osmnx/blob/master/osmnx/simplify.py
def is_endpoint(G, node, strict=True):
    """
    Return True if the node is a "real" endpoint of an edge in the network, \
    otherwise False. OSM data includes lots of nodes that exist only as points \
    to help streets bend around curves. An end point is a node that either: \
    1) is its own neighbor, ie, it self-loops. \
    2) or, has no incoming edges or no outgoing edges, ie, all its incident \
        edges point inward or all its incident edges point outward. \
    3) or, it does not have exactly two neighbors and degree of 2 or 4. \
    4) or, if strict mode is false, if its edges have different OSM IDs. \
    Parameters
    ----------
    G : networkx multidigraph
    node : int
        the node to examine
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules \
        but have edges with different OSM IDs
    Returns
    -------
    bool
    """
    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    if node in neighbors:
        # if the node appears in its list of neighbors, it self-loops. this is
        # always an endpoint.
        return True

    # if node has no incoming edges or no outgoing edges, it must be an endpoint
    elif G.out_degree(node)==0 or G.in_degree(node)==0:
        return True

    elif not (n==2 and (d==2 or d==4)):
        # else, if it does NOT have 2 neighbors AND either 2 or 4 directed
        # edges, it is an endpoint. either it has 1 or 3+ neighbors, in which
        # case it is a dead-end or an intersection of multiple streets or it has
        # 2 neighbors but 3 degree (indicating a change from oneway to twoway)
        # or more than 4 degree (indicating a parallel edge) and thus is an
        # endpoint
        return True

    elif not strict:
        # non-strict mode
        osmids = []

        # add all the edge OSM IDs for incoming edges
        for u in G.predecessors(node):
            for key in G[u][node]:
                osmids.append(G.edges[u, node, key]['osmid'])

        # add all the edge OSM IDs for outgoing edges
        for v in G.successors(node):
            for key in G[node][v]:
                osmids.append(G.edges[node, v, key]['osmid'])

        # if there is more than 1 OSM ID in the list of edge OSM IDs then it is
        # an endpoint, if not, it isn't
        return len(set(osmids)) > 1

    else:
        # if none of the preceding rules returned true, then it is not an endpoint
        return False


# https://github.com/gboeing/osmnx/blob/master/osmnx/simplify.py
def build_path(G, node, endpoints, path):
    """
    Recursively build a path of nodes until you hit an endpoint node.
    Parameters
    ----------
    G : networkx multidigraph
    node : int
        the current node to start from
    endpoints : set
        the set of all nodes in the graph that are endpoints
    path : list
        the list of nodes in order in the path so far
    Returns
    -------
    paths_to_simplify : list
    """
    # for each successor in the passed-in node
    for successor in G.successors(node):
        if successor not in path:
            # if this successor is already in the path, ignore it, otherwise add
            # it to the path
            path.append(successor)
            if successor not in endpoints:
                # if this successor is not an endpoint, recursively call
                # build_path until you find an endpoint
                path = build_path(G, successor, endpoints, path)
            else:
                # if this successor is an endpoint, we've completed the path,
                # so return it
                return path

    if (path[-1] not in endpoints) and (path[0] in G.successors(path[-1])):
        # if the end of the path is not actually an endpoint and the path's
        # first node is a successor of the path's final node, then this is
        # actually a self loop, so add path's first node to end of path to
        # close it
        path.append(path[0])

    return path


# https://github.com/gboeing/osmnx/blob/master/osmnx/simplify.py
def get_paths_to_simplify(G, strict=True, verbose=False):
    """
    Create a list of all the paths to be simplified between endpoint nodes.
    The path is ordered from the first endpoint, through the interstitial nodes,
    to the second endpoint. If your street network is in a rural area with many
    interstitial nodes between true edge endpoints, you may want to increase
    your system's recursion limit to avoid recursion errors.
    Parameters
    ----------
    G : networkx multidigraph
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs
    Returns
    -------
    paths_to_simplify : list
    """

    # first identify all the nodes that are endpoints
    start_time = time.time()
    endpoints = set([node for node in G.nodes() if is_endpoint(G, node, strict=strict)])
    if verbose:
        print('Identified {:,} edge endpoints in {:,.2f} seconds'.format(len(endpoints), time.time()-start_time))

    start_time = time.time()
    paths_to_simplify = []

    # for each endpoint node, look at each of its successor nodes
    for node in endpoints:
        for successor in G.successors(node):
            if successor not in endpoints:
                # if the successor is not an endpoint, build a path from the
                # endpoint node to the next endpoint node
                try:
                    path = build_path(G, successor, endpoints, path=[node, successor])
                    paths_to_simplify.append(path)
                except RuntimeError:
                    print('Recursion error: exceeded max depth, moving on to next endpoint successor', level=lg.WARNING)
                    # recursion errors occur if some connected component is a
                    # self-contained ring in which all nodes are not end points.
                    # could also occur in extremely long street segments (eg, in
                    # rural areas) with too many nodes between true endpoints.
                    # handle it by just ignoring that component and letting its
                    # topology remain intact (this should be a rare occurrence)
                    # RuntimeError is what Python <3.5 will throw, Py3.5+ throws
                    # RecursionError but it is a subtype of RuntimeError so it
                    # still gets handled

    if verbose:
        print('Constructed all paths to simplify in {:,.2f} seconds'.format(time.time()-start_time))
    return paths_to_simplify


# https://github.com/gboeing/osmnx/blob/master/osmnx/simplify.py
def is_simplified(G):
    """
    Determine if a graph has already had its topology simplified.
    If any of its edges have a geometry attribute, we know that it has
    previously been simplified.
    Parameters
    ----------
    G : networkx multidigraph
    Returns
    -------
    bool
    """

    return 'simplified' in G.graph and G.graph['simplified']


# https://github.com/gboeing/osmnx/blob/master/osmnx/simplify.py
def simplify_graph(G, strict=True, verbose=False):
    """
    Simplify a graph's topology by removing all nodes that are not intersections
    or dead-ends.
    Create an edge directly between the end points that encapsulate them,
    but retain the geometry of the original edges, saved as attribute in new
    edge.
    Parameters
    ----------
    G : networkx multidigraph
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs
    Returns
    -------
    networkx multidigraph
    """

    if is_simplified(G):
        raise Exception('This graph has already been simplified, cannot simplify it again.')

    if verbose:
        print('Begin topologically simplifying the graph...')
    G = G.copy()
    initial_node_count = len(list(G.nodes()))
    initial_edge_count = len(list(G.edges()))
    all_nodes_to_remove = []
    all_edges_to_add = []

    # construct a list of all the paths that need to be simplified
    paths = get_paths_to_simplify(G, strict=strict)

    start_time = time.time()
    for path in paths:

        # add the interstitial edges we're removing to a list so we can retain
        # their spatial geometry
        edge_attributes = {}
        for u, v in zip(path[:-1], path[1:]):

            # there shouldn't be multiple edges between interstitial nodes
            if not G.number_of_edges(u, v) == 1:
                if verbose:
                    print('Multiple edges between "{}" and "{}" found when simplifying'.format(u, v), level=lg.WARNING)

            # the only element in this list as long as above check is True
            # (MultiGraphs use keys (the 0 here), indexed with ints from 0 and
            # up)
            # print(G.edges([u,v]))
            try:
                edge = G.edges[u, v, 0]
            except:
                edge = G.edges[u, v, "0"]
                
            for key in edge:
                if key in edge_attributes:
                    # if this key already exists in the dict, append it to the
                    # value list
                    edge_attributes[key].append(edge[key])
                else:
                    # if this key doesn't already exist, set the value to a list
                    # containing the one value
                    edge_attributes[key] = [edge[key]]

        for key in edge_attributes:
            # print("key", key, "edge_attributes:", edge_attributes)
            # print(" ", edge_attributes[key])
            # don't touch the length attribute, we'll sum it at the end
            if len(set(edge_attributes[key])) == 1 and not key == 'length':
                # if there's only 1 unique value in this attribute list,
                # consolidate it to the single value (the zero-th)
                edge_attributes[key] = edge_attributes[key][0]
            elif not key == 'length':
                # otherwise, if there are multiple values, keep one of each value
                edge_attributes[key] = list(set(edge_attributes[key]))

        # construct the geometry and sum the lengths of the segments
        edge_attributes['geometry'] = LineString([Point((G.nodes[node]['x'], G.nodes[node]['y'])) for node in path])
        edge_attributes['length'] = sum(edge_attributes['length'])

        # add the nodes and edges to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append({'origin':path[0],
                                 'destination':path[-1],
                                 'attr_dict':edge_attributes})

    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge['origin'], edge['destination'], **edge['attr_dict'])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    G.graph['simplified'] = True

    msg = 'Simplified graph (from {:,} to {:,} nodes and from {:,} to {:,} edges) in {:,.2f} seconds'
    if verbose:
        print(msg.format(initial_node_count, len(list(G.nodes())), initial_edge_count, len(list(G.edges())), time.time()-start_time))
    return G


def make_str(value):
    """
    Convert a passed-in value to unicode if Python 2, or string if Python 3.
    Parameters
    ----------
    value : any
        the value to convert to unicode/string
    Returns
    -------
    unicode or string
    """
    try:
        # for python 2.x compatibility, use unicode
        return unicode(value)
    except NameError:
        # python 3.x has no unicode type, so if error, use str type
        return str(value)
    
    
# https://github.com/gboeing/osmnx/blob/master/osmnx/simplify.py
def clean_intersections(G, tolerance=15, dead_ends=False):
    """
    Clean-up intersections comprising clusters of nodes by merging them and
    returning their centroids.
    Divided roads are represented by separate centerline edges. The intersection
    of two divided roads thus creates 4 nodes, representing where each edge
    intersects a perpendicular edge. These 4 nodes represent a single
    intersection in the real world. This function cleans them up by buffering
    their points to an arbitrary distance, merging overlapping buffers, and
    taking their centroid. For best results, the tolerance argument should be
    adjusted to approximately match street design standards in the specific
    street network.
    Parameters
    ----------
    G : networkx multidigraph
    tolerance : float
        nodes within this distance (in graph's geometry's units) will be
        dissolved into a single intersection
    dead_ends : bool
        if False, discard dead-end nodes to return only street-intersection
        points
    Returns
    ----------
    intersection_centroids : geopandas.GeoSeries
        a GeoSeries of shapely Points representing the centroids of street
        intersections
    """

    # if dead_ends is False, discard dead-end nodes to only work with edge
    # intersections
    if not dead_ends:
        if 'streets_per_node' in G.graph:
            streets_per_node = G.graph['streets_per_node']
        else:
            streets_per_node = 1    # count_streets_per_node(G)

        dead_end_nodes = [node for node, count in streets_per_node.items() if count <= 1]
        G = G.copy()
        G.remove_nodes_from(dead_end_nodes)

    # create a GeoDataFrame of nodes, buffer to passed-in distance, merge
    # overlaps
    gdf_nodes = graph_to_gdfs(G, edges=False)
    buffered_nodes = gdf_nodes.buffer(tolerance).unary_union
    if isinstance(buffered_nodes, Polygon):
        # if only a single node results, make it iterable so we can turn it
        # int a GeoSeries
        buffered_nodes = [buffered_nodes]

    # get the centroids of the merged intersection polygons
    unified_intersections = gpd.GeoSeries(list(buffered_nodes))
    intersection_centroids = unified_intersections.centroid
    return intersection_centroids


def great_circle_vec(lat1, lng1, lat2, lng2, earth_radius=6371009):
    """
    https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py
    Vectorized function to calculate the great-circle distance between two
    points or between vectors of points, using haversine.
    Parameters
    ----------
    lat1 : float or array of float
    lng1 : float or array of float
    lat2 : float or array of float
    lng2 : float or array of float
    earth_radius : numeric
        radius of earth in units in which distance will be returned (default is
        meters)
    Returns
    -------
    distance : float or vector of floats
        distance or vector of distances from (lat1, lng1) to (lat2, lng2) in
        units of earth_radius
    """

    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    d_phi = phi2 - phi1

    theta1 = np.deg2rad(lng1)
    theta2 = np.deg2rad(lng2)
    d_theta = theta2 - theta1

    h = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_theta / 2) ** 2
    h = np.minimum(1.0, h)  # protect against floating point errors

    arc = 2 * np.arcsin(np.sqrt(h))

    # return distance in units of earth_radius
    distance = arc * earth_radius
    return distance


def induce_subgraph(G, node_subset):
    """
    https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py
    Induce a subgraph of G.
    Parameters
    ----------
    G : networkx multidigraph
    node_subset : list-like
        the subset of nodes to induce a subgraph of G
    Returns
    -------
    G2 : networkx multidigraph
        the subgraph of G induced by node_subset
    """

    node_subset = set(node_subset)

    # copy nodes into new graph
    G2 = G.__class__()
    G2.add_nodes_from((n, G.nodes[n]) for n in node_subset)

    # copy edges to new graph, including parallel edges
    if G2.is_multigraph:
        G2.add_edges_from((n, nbr, key, d)
            for n, nbrs in G.adj.items() if n in node_subset
            for nbr, keydict in nbrs.items() if nbr in node_subset
            for key, d in keydict.items())
    else:
        G2.add_edges_from((n, nbr, d)
            for n, nbrs in G.adj.items() if n in node_subset
            for nbr, d in nbrs.items() if nbr in node_subset)

    # update graph attribute dict, and return graph
    G2.graph.update(G.graph)
    return G2


def get_largest_component(G, strongly=False, verbose=False):
    """
    https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py
    Return a subgraph of the largest weakly or strongly connected component
    from a directed graph.
    Parameters
    ----------
    G : networkx multidigraph
    strongly : bool
        if True, return the largest strongly instead of weakly connected
        component
    Returns
    -------
    G : networkx multidigraph
        the largest connected component subgraph from the original graph
    """

    start_time = time.time()
    original_len = len(list(G.nodes()))

    if strongly:
        # if the graph is not connected retain only the largest strongly connected component
        if not nx.is_strongly_connected(G):

            # get all the strongly connected components in graph then identify the largest
            sccs = nx.strongly_connected_components(G)
            largest_scc = max(sccs, key=len)
            G = induce_subgraph(G, largest_scc)

            msg = ('Graph was not connected, retained only the largest strongly '
                   'connected component ({:,} of {:,} total nodes) in {:.2f} seconds')
            if verbose:
                print(msg.format(len(list(G.nodes())), original_len, time.time()-start_time))
    else:
        # if the graph is not connected retain only the largest weakly connected component
        if not nx.is_weakly_connected(G):

            # get all the weakly connected components in graph then identify the largest
            wccs = nx.weakly_connected_components(G)
            largest_wcc = max(wccs, key=len)
            G = induce_subgraph(G, largest_wcc)

            msg = ('Graph was not connected, retained only the largest weakly '
                   'connected component ({:,} of {:,} total nodes) in {:.2f} seconds')
            if verbose:
                print(msg.format(len(list(G.nodes())), original_len, time.time()-start_time))

    return G


def add_path(G, data, one_way):
    """
    https://github.com/gboeing/osmnx/blob/master/osmnx/core.py
    Add a path to the graph.
    Parameters
    ----------
    G : networkx multidigraph
    data : dict
        the attributes of the path
    one_way : bool
        if this path is one-way or if it is bi-directional
    Returns
    -------
    None
    """

    # extract the ordered list of nodes from this path element, then delete it
    # so we don't add it as an attribute to the edge later
    path_nodes = data['nodes']
    del data['nodes']

    # set the oneway attribute to the passed-in value, to make it consistent
    # True/False values
    data['oneway'] = one_way

    # zip together the path nodes so you get tuples like (0,1), (1,2), (2,3)
    # and so on
    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
    G.add_edges_from(path_edges, **data)

    # if the path is NOT one-way
    if not one_way:
        # reverse the direction of each edge and add this path going the
        # opposite direction
        path_edges_opposite_direction = [(v, u) for u, v in path_edges]
        G.add_edges_from(path_edges_opposite_direction, **data)


def add_paths(G, paths, bidirectional=False):
    """
    https://github.com/gboeing/osmnx/blob/master/osmnx/core.py
    Add a collection of paths to the graph.
    Parameters
    ----------
    G : networkx multidigraph
    paths : dict
        the paths from OSM
    bidirectional : bool
        if True, create bidirectional edges for one-way streets
    Returns
    -------
    None
    """

    # the list of values OSM uses in its 'oneway' tag to denote True
    osm_oneway_values = ['yes', 'true', '1', '-1']

    for data in paths.values():

        # if this path is tagged as one-way and if it is not a walking network,
        # then we'll add the path in one direction only
        if ('oneway' in data and data['oneway'] in osm_oneway_values) and not bidirectional:
            if data['oneway'] == '-1':
                # paths with a one-way value of -1 are one-way, but in the
                # reverse direction of the nodes' order, see osm documentation
                data['nodes'] = list(reversed(data['nodes']))
            # add this path (in only one direction) to the graph
            add_path(G, data, one_way=True)

        elif ('junction' in data and data['junction'] == 'roundabout') and not bidirectional:
            # roundabout are also oneway but not tagged as is
            add_path(G, data, one_way=True)

        # else, this path is not tagged as one-way or it is a walking network
        # (you can walk both directions on a one-way street)
        else:
            # add this path (in both directions) to the graph and set its
            # 'oneway' attribute to False. if this is a walking network, this
            # may very well be a one-way street (as cars/bikes go), but in a
            # walking-only network it is a bi-directional edge
            add_path(G, data, one_way=False)

    return G


def add_edge_lengths(G, verbose=False):
    """
    https://github.com/gboeing/osmnx/blob/master/osmnx/core.py
    Add length (meters) attribute to each edge by great circle distance between
    nodes u and v.
    Parameters
    ----------
    G : networkx multidigraph
    Returns
    -------
    G : networkx multidigraph
    """

    start_time = time.time()

    # first load all the edges' origin and destination coordinates as a
    # dataframe indexed by u, v, key
    coords = np.array([[u, v, k, G.nodes[u]['y'], G.nodes[u]['x'], G.nodes[v]['y'], G.nodes[v]['x']] for u, v, k in G.edges(keys=True)])
    df_coords = pd.DataFrame(coords, columns=['u', 'v', 'k', 'u_y', 'u_x', 'v_y', 'v_x'])
    df_coords[['u', 'v', 'k']] = df_coords[['u', 'v', 'k']].astype(np.int64)
    df_coords = df_coords.set_index(['u', 'v', 'k'])

    # then calculate the great circle distance with the vectorized function
    gc_distances = great_circle_vec(lat1=df_coords['u_y'],
                                    lng1=df_coords['u_x'],
                                    lat2=df_coords['v_y'],
                                    lng2=df_coords['v_x'])

    # fill nulls with zeros and round to the millimeter
    gc_distances = gc_distances.fillna(value=0).round(3)
    nx.set_edge_attributes(G, name='length', values=gc_distances.to_dict())

    if  verbose:
        print('Added edge lengths to graph in {:,.2f} seconds'.format(time.time()-start_time))
    return G
