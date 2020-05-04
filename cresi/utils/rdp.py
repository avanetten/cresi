#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code borrowd from:
https://github.com/mitroadmaps/roadtracer/blob/master/lib/discoverlib/rdp.py

The Ramer-Douglas-Peucker algorithm roughly ported from the pseudo-code provided
by http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
"""

from math import sqrt

def distance(a, b):
    return  sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
    if (start == end):
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) - (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d

def rdp(points, epsilon=1):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results


#def simplify_graph(graph, max_distance=1):
#    """
#    https://github.com/anilbatra2185/road_connectivity/blob/master/data_utils/graph_utils.py
#    @params graph: MultiGraph object of networkx
#    @return: simplified graph after applying RDP algorithm.
#    """
#    all_segments = []
#    # Iterate over Graph Edges
#    for (s, e) in graph.edges():
#        for _, val in graph[s][e].items():
#            # get all pixel points i.e. (x,y) between the edge
#            ps = val["pts"]
#            # create a full segment
#            full_segments = np.row_stack([graph.node[s]["o"], ps, graph.node[e]["o"]])
#            # simply the graph.
#            segments = rdp.rdp(full_segments.tolist(), max_distance)
#            all_segments.append(segments)
#
#    return all_segments