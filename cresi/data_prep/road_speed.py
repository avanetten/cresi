#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:30:30 2019

@author: avanetten

https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=17036&pm=14735
https://arxiv.org/pdf/1807.01232.pdf


Appendix A: The SpaceNet Roads Dataset Labeling Guidelines
The SpaceNet Roads Dataset labeling guidelines: 

1.	Road vectors must be drawn as a center line within 2m (7 pixels) of observed road
a.	The centerline of a road is defined as the centerline of the roadway. If a road has an even number of lanes, the centerline shall be drawn on the line separating lanes.  If the road has an odd number of lanes then the centerline should be drawn down the center of the middle lane.  
b.	Divided highways should have two centerlines, a centerline for each direction of traffic.  See below for the definition of a divided highway. 
2.	Road vectors must be represented as a connected network to support routing.  Roads that intersect each other should share points as an intersection like instructed through OSM.  Roads that cross each other that are not connected such as while using an overpass should not share a point of connection.
3.	Roads must not bisect building footprints.
4.	Sections of a road that are a bridge or overpass must be labeled as a bridge via a Boolean flag.  
5.	Divided highways must be represented as two lines with traffic direction indicated when possible.
6.	 Surface type must be classified as:  paved, unpaved, or unknown. 
7.	Road will be identified by type: (Motorway, Primary, Secondary, Tertiary, Residential, Unclassified, Cart Track)
8.	Number of lanes will be listed as number of lanes for each centerline as defined in rule 1.  If road has two lanes in each direction, the number of lanes shall equal 4.  If a road has 3 lanes in one direction and 2 directions in another the total number of lanes shall be 5.  


Definition of Divided Highway: 

A divided highway is a road that has a median or barrier that physically prevents turns across traffic.  
A median can be:
●	Concrete
●	Asphalt
●	Green Space
●	Dirt/unpaved road
A median is not:
●	Yellow hatched lines on pavement.  


Road Type Guidelines: 
All road types were defined using the Open Street Maps taxonomy for key=highway. The below descriptions were taken from the Open Street Maps tagging guidelines for highway and the East Africa Tagging Guidelines.    
1.	motorway - A restricted access major divided highway, normally with 2 or more running lanes plus emergency hard shoulder. Access onto a motorway comes exclusively through ramps (controlled access).  Equivalent to the Freeway, Autobahn, etc.
2.	primary - National roads connect the most important cities/towns in a country. In most countries, these roads will usually be tarmacked and show center markings. (In South Sudan, however, primary roads might also be unpaved.)
3.	secondary – Secondary roads are the second most important roads in a country's transport system. They typically link medium-sized places. They may be paved but in in some countries they are not.
4.	tertiary - Tertiary roads are busy through roads that link smaller towns and larger villages. More often than not, these roads will be unpaved. However, this tag should be used only on roads wide enough to allow two cars to pass safely.
5.	residential - Roads which serve as an access to housing, without function of connecting settlements. Often lined with housing.
6.	unclassified -The least important through roads in a country's system – i.e. minor roads of a lower classification than tertiary, but which serve a purpose other than access to properties. Often link villages and hamlets. (The word 'unclassified' is a historical artifact of the UK road system and does not mean that the classification is unknown; you can use highway=road for that.)
7.	Cart track – This is a dirt path that shows vehicle traffic that is less defined than a residential   

Additional information and rules for road identification come from following sources:
1: http://wiki.openstreetmap.org/wiki/Highway_Tag_Africa
2: http://wiki.openstreetmap.org/wiki/East_Africa_Tagging_Guidelines
3: http://wiki.openstreetmap.org/wiki/Key:highway


GeoJSON Schema 
Attributes:
1)	“geometry”: Linestring
2)	“road_id”: int 
Identifier Index
3)	“road_type”: int
1: Motorway
2: Primary
3: Secondary
4: Tertiary
5: Residential
6: Unclassified
7: Cart track
4)	“paved”: int
1: Paved
2: Unpaved
3: Unknown
5)	“bridge_typ”: int
1: Bridge
2: Not a bridge
3: Unknown
6)	“lane_number”: int
1: one lane
2: two lanes
3: three lanes
etc.


# geojson example
{ "type": "Feature", "properties": { "gid": 15806, "road_id": 24791, "road_type": 5, "paved": 1, "bridge": 2, "one_way": 2, "heading": 0.0, "lane_numbe": 2, "ingest_tim": "2017\/09\/24 20:36:06.436+00", "edit_date": "2017\/09\/24 20:36:06.436+00", "edit_user": "ian_kitchen", "production": "0", "imagery_so": "0", "imagery_da": "0", "partialBuilding": 1.0, "partialDec": 0.0 }, "geometry": { "type": "LineString", "coordinates": [ [ -115.305975139809291, 36.179169421086783, 0.0 ], [ -115.305540626738249, 36.179686396492464, 0.0 ], [ -115.305150516462803, 36.180003559318038, 0.0 ], [ -115.304760406187356, 36.18037781145221, 0.0 ], [ -115.304287833577249, 36.180932846396956, 0.0 ], [ -115.304305558679488, 36.18094769983459, 0.0 ] ] } }
"""

import os
import sys
import json
import math
import fiona
import shutil
import shapely
import numpy as np
import geopandas as gpd
import random
import skimage
import argparse
import cv2
from osgeo import gdal, ogr, osr
from json import JSONDecodeError
random.seed(2019)

# import relative paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
import osmnx_funcs


###############################################################################
def speed_func(geojson_row, label_type='sn5', verbose=True):
    '''
    Infer road speed limit based on SpaceNet properties
    # geojson example
    { "type": "Feature", "properties": { "gid": 15806, "road_id": 24791,
            "road_type": 5, "paved": 1, "bridge": 2, "one_way": 2,
            "heading": 0.0, "lane_numbe": 2,
            "ingest_tim": "2017\/09\/24 20:36:06.436+00",
            "edit_date": "2017\/09\/24 20:36:06.436+00",
            "edit_user": "ian_kitchen", "production": "0", "imagery_so": "0",
            "imagery_da": "0", "partialBuilding": 1.0, "partialDec": 0.0 }, 
            "geometry": { "type": "LineString", "coordinates": [ [ -115.305975139809291, 36.179169421086783, 0.0 ], [ -115.305540626738249, 36.179686396492464, 0.0 ], [ -115.305150516462803, 36.180003559318038, 0.0 ], [ -115.304760406187356, 36.18037781145221, 0.0 ], [ -115.304287833577249, 36.180932846396956, 0.0 ], [ -115.304305558679488, 36.18094769983459, 0.0 ] ] } 
    }
    label_type = ['sn5', 'sn3', 'osm']
    if osm_labels, assume an osm label
    Return speed_final_mph, speed_final_mps
    Set < 0 if not found
    '''

    keys = set(geojson_row['properties'].keys())
    # print ("geojson_row:", geojson_row)
    if verbose:
        print("geojson_row", geojson_row)

    if label_type == 'osm':
        # https://wiki.openstreetmap.org/wiki/Key:highway
        # convert road_type_str to int
        conv_dict = {'motorway': 1,
                     'motorway_link': 1,
                     'trunk': 1,
                     'trunk_link': 1,
                     'primary': 2,
                     'primary_link': 2,
                     'bus_guideway': 2,
                     'secondary': 3,
                     'secondary_link': 3,
                     'tertiary': 4,
                     'tertiary_link': 4,
                     'residential': 5,
                     'living_street': 5,
                     'unclassified': 6,
                     'service': 6,
                     'road': 6,
                     'cart_track': 7,
                     'track': 7,
                     }
        skip_set = ('stopline', 'footway', 'bridleway', 'step', 'steps', 
                    'path', 'pedestrian', 'escape', 'cycleway', 'raceway',
                    'bus', 'services', 'escalator', 'sidewalk')


        if ('highway' in geojson_row['properties']):
            road_type_str = geojson_row['properties']['highway']
        elif ('class' in geojson_row['properties']):
            if geojson_row['properties']['class'] == 'highway':
                road_type_str = geojson_row['properties']['type']
            else:
                if verbose:
                    print("  class not highway")
                return -1, -1
        else:
            if verbose:
                print("  not road")
            return -1, -1

        # check type
        if road_type_str in skip_set:
            if verbose:
                print("road_type {} in skip_set".format(road_type_str))
            # print ("  geojson_row['properties']['highway']:", geojson_row['properties']['highway'])
            return -1, -1
        else:
            # print ("  geojson_row['properties']['highway']:", geojson_row['properties']['highway'])
            if verbose:
                print("  road_type_str:", road_type_str)
            road_type = conv_dict[road_type_str.lower()]
            
        # check if tunnel
        if 'tunnel' in keys:
            if geojson_row['properties']['tunnel'] in ['yes']:
                print ("  skipping tunnel!", geojson_row)
                return -1, -1

        num_lanes = 2
        surface = 1
        bridge = 2

    # SpaceNet 3 labels
    elif label_type == 'sn3':
        # sometimes we get a malformed road
        try:
            road_type = int(geojson_row['properties']['road_type'])
            # lane number was incorrectly labeled in initial geojsons
            if 'lane_numbe' in keys:
                num_lanes = int(geojson_row['properties']['lane_numbe'])
            else:
                num_lanes = int(geojson_row['properties']['lane_number'])
            surface = int(geojson_row['properties']['paved'])
            if 'bridge_typ' in keys:
                bridge = int(geojson_row['properties']['bridge_typ'])
            else:
                bridge = int(geojson_row['properties']['bridge_type'])
        except KeyError:
            # assume a one lane unclassified paved road
            road_type = 6
            num_lanes = 1
            surface = 1
            bridge = 2
            print("malformed geojson row:", geojson_row)

    # SpaceNet 5 labels
    elif label_type == 'sn5':
        # e.g., { "type": "Feature", "properties": { "OBJECTID": "43", "bridge": null, "highway": "unclassified", "osm_id": 683685519.000000, "surface": "paved", "lanes": "2" }, "geometry": { "type": "LineString", "coordinates": [ [ 37.633999, 55.626647500000047 ], [ 37.633841700000062, 55.625982300000032 ], [ 37.633794500000079, 55.625240800000029 ], [ 37.633748600000047, 55.625123100000053 ], [ 37.633341800000039, 55.624770700000056 ] ] } },

        highway_conv_dict = {'motorway': 1,
                     'motorway_link': 1,
                     'trunk': 1,
                     'trunk_link': 1,
                     'primary': 2,
                     'primary_link': 2,
                     'bus_guideway': 2,
                     'secondary': 3,
                     'secondary_link': 3,
                     'tertiary': 4,
                     'tertiary_link': 4,
                     'residential': 5,
                     'living_street': 5,
                     'unclassified': 6,
                     'service': 6,
                     'road': 6,
                     'cart_track': 7,
                     'track': 7,
                     }
        # road_type
        road_type_str = geojson_row['properties']['highway']
        road_type = highway_conv_dict[road_type_str.lower()]
        
        # bridge
        bridge_str = geojson_row['properties']['bridge']
        if bridge_str == 'null':
            bridge = 2
        else:
            bridge = 1

        # surface type
        surface_str = geojson_row['properties']['surface']
        if surface_str == 'paved':
            surface = 1
        else:
            surface = 2
        
        # num lanes
        num_lanes = int(float(geojson_row['properties']['lanes']))
        
        if verbose:
            print("road_type:", road_type)
            print("surface:", surface)
            print("num_lanes:", num_lanes)


    # road type (int)
    '''
    1: Motorway
    2: Primary
    3: Secondary
    4: Tertiary
    5: Residential
    6: Unclassified
    7: Cart track
    '''
    #road_type_dict = {
    #    1: 60,
    #    2: 45,
    #    3: 35,
    #    4: 25,
    #    5: 25,
    #    6: 20,
    #    7: 15
    #}

    # https://en.wikipedia.org/wiki/File:Speed_limits_in_Ohio.svg
    # https://wiki.openstreetmap.org/wiki/OSM_tags_for_routing/Maxspeed#United_States_of_America    
    #   Use Oregon:
    #       State	Motorway	Trunk	Primary	Secondary	Tertiary	Unclassified	Residential	Living street	Service
    #       Oregon	55 mph	55 mph	55 mph	35 mph	30 mph		         25 mph		              15 mph
    # feed in [road_type][num_lanes]
    nested_speed_dict = {
        1: {1: 55, 2: 55, 3: 65, 4: 65, 5: 65, 6: 65, 7: 65, 8: 65, 9: 65, 10: 65, 11: 65, 12: 65},
        2: {1: 45, 2: 45, 3: 55, 4: 55, 5: 55, 6: 55, 7: 55, 8: 55, 9: 55, 10: 55, 11: 55, 12: 55},
        3: {1: 35, 2: 35, 3: 45, 4: 45, 5: 45, 6: 45, 7: 45, 8: 45, 9: 45, 10: 45, 11: 45, 12: 45},
        4: {1: 30, 2: 30, 3: 35, 4: 35, 5: 35, 6: 35, 7: 35, 8: 35, 9: 35, 10: 35, 11: 35, 12: 35},
        5: {1: 25, 2: 25, 3: 30, 4: 30, 5: 30, 6: 30, 7: 30, 8: 30, 9: 30, 10: 30, 11: 30, 12: 30},
        6: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20, 11: 20, 12: 20},
        7: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20, 11: 20, 12: 20}
    }

    # multiply speed by this factor based on surface
    # 1 = paved, 2 = unpaved
    road_surface_dict = {
        1: 1,
        2: 0.75
    }

    # multiply speed by this factor for bridges
    # 1 = bridge, 2 = not bridge
    bridge_dict = {
        1: 1,  # 0.8,
        2: 1}

#    # V0, Feb 22 2019 and prior
#    # feed in [road_type][num_lanes]
#    nested_speed_dict = {
#        1: {1: 45, 2: 50, 3: 55, 4: 65, 5: 65, 6: 65, 7: 65, 8: 65, 9: 65, 10: 65, 11: 65, 12: 65},
#        2: {1: 35, 2: 40, 3: 45, 4: 45, 5: 45, 6: 45, 7: 45, 8: 45, 9: 45, 10: 45, 11: 45, 12: 45},
#        3: {1: 30, 2: 30, 3: 30, 4: 30, 5: 30, 6: 30, 7: 30, 8: 30, 9: 30, 10: 30, 11: 30, 12: 30},
#        4: {1: 25, 2: 25, 3: 25, 4: 25, 5: 25, 6: 25, 7: 25, 8: 25, 9: 25, 10: 25, 11: 25, 12: 25},
#        5: {1: 25, 2: 25, 3: 25, 4: 25, 5: 25, 6: 25, 7: 25, 8: 25, 9: 25, 10: 25, 11: 25, 12: 25},
#        6: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20, 11: 20, 12: 20},
#        7: {1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 20, 11: 20, 12: 20}
#    }
#    # multiply speed by this factor based on surface
#    road_surface_dict = {
#        1: 1,
#        2: 0.75
#    }
#    # multiply speed by this factor for bridges
#    bridge_dict = {
#        1: 1, #0.8,
#        2: 1}

    # default speed in miles per hour
    speed_init_mph = nested_speed_dict[road_type][num_lanes]
    # reduce speed for unpaved or bridge
    speed_final_mph = speed_init_mph * road_surface_dict[surface] \
        * bridge_dict[bridge]
    # get speed in meters per second
    speed_final_mps = 0.44704 * speed_final_mph

    if verbose:
        print("speed_mph:", speed_final_mph)

    return speed_final_mph, speed_final_mps


###############################################################################
def speed_to_burn_val(speed, min_speed=15, max_speed=65.,
                      min_road_burn_val=127,
                      mask_max=255):
    '''Convert speed estimate to mask burn value between 0 and 255'''

    bw = mask_max - min_road_burn_val
    burn_val = min_road_burn_val + bw \
        * ((speed - min_speed) / (max_speed - min_speed))
    return max(burn_val, min_road_burn_val)


###############################################################################
def speed_to_bins_bg(speed_mph, bin_size_mph=10.0, channel_value_mult=1.):
    '''Convert speed estimate to appropriate channel
    bin = 0 if speed = 0'''

    return int(int(math.ceil(speed_mph / bin_size_mph)) * channel_value_mult)


###############################################################################
def update_feature_name(geojson_row, name_bad, name_good):
    # optional: also update "ingest_tim" tag
    if name_bad in geojson_row['properties'].keys():
        x_tmp = geojson_row['properties'][name_bad]
        del geojson_row['properties'][name_bad]
        geojson_row['properties'][name_good] = x_tmp
    return


###############################################################################
def add_speed_to_geojson(geojson_path_in, geojson_path_out,
                         label_type='sn5',
                         speed_key_mph='inferred_speed_mph',
                         speed_key_mps='inferred_speed_mps',
                         verbose=True):
    '''Update geojson data to add inferred speed information'''

    speed_mph_set = set()
    speed_mph_arr = []
    bad_row_idxs = []
    with open(geojson_path_in, 'r+') as f:
        try:
            geojson_data = json.load(f)
        except JSONDecodeError:
            # assume empty array, copy
            shutil.copy(geojson_path_in, geojson_path_out)
            return [], set()

        init_len = len(geojson_data['features'])
        for i, geojson_row in enumerate(geojson_data['features']):
            if verbose and (i % 100) == 0:
                print("\n", i, "/", init_len, "geojson_row:", geojson_row)

            # optional: also update "ingest_tim" tag
            if 'ingest_tim' in geojson_row['properties'].keys():
                x_tmp = geojson_row['properties']['ingest_tim']
                del geojson_row['properties']['ingest_tim']
                geojson_row['properties']['ingest_time'] = x_tmp

            # optional: also update "bridge_typ" tag
            if 'bridge_typ' in geojson_row['properties'].keys():
                x_tmp = geojson_row['properties']['bridge_typ']
                del geojson_row['properties']['bridge_typ']
                geojson_row['properties']['bridge_type'] = x_tmp

            # optional: also update "lane_numbe" tag
            if 'lane_numbe' in geojson_row['properties'].keys():
                x_tmp = geojson_row['properties']['lane_numbe']
                del geojson_row['properties']['lane_numbe']
                geojson_row['properties']['lane_number'] = x_tmp

            # infer route speed limit
            speed_mph, speed_mps = speed_func(geojson_row, 
                                              label_type=label_type,
                                              verbose=verbose)
            if verbose:
                print("  speed_mph, speed_mps:", speed_mph, speed_mps)
            if speed_mph >= 0:
                # update properties
                geojson_row['properties'][speed_key_mph] = speed_mph
                geojson_row['properties'][speed_key_mps] = speed_mps
                speed_mph_set.add(speed_mph)
                speed_mph_arr.append(speed_mph)
            else:
                if verbose:
                    print("geojson_row:", geojson_row)
                bad_row_idxs.append(i)

    # remove bad idxs
    if len(bad_row_idxs) > 0:
        # geojson_data['features'] = [e+'\n' for i, e in
        geojson_data['features'] = [e+'\n' for i, e in
                enumerate(geojson_data['features']) if i not in bad_row_idxs]
        final_len = len(geojson_data['features'])
        # print("  bad_row_idxs:", bad_row_idxs)
        if 2 > 1: #verbose:
            print("  init_len:", init_len, "final_len:", final_len)

    # save file
    with open(geojson_path_out, 'w') as f:
        f.write(json.dumps(geojson_data, indent=2))
        if verbose:
            print("geojson_path_out:", geojson_path_out)
        # f.write(json.dumps(geojson_data, indent=1))

#    # older version that doesn't print correctly
#    geojson_data = fiona.open(geojson_path, 'r')
#    out = []
#    for i,geojson_row in enumerate(geojson_data):
#        if verbose:
#            print ("\ngeojson_row:", geojson_row)
#        # infer route speed limit
#        speed_mph, speed_mps = speed_func(geojson_row)
#        if verbose:
#            print ("  speed_mph, speed_mps:", speed_mph, speed_mps)
#        # update properties
#        geojson_row['properties']['speed_mph'] = speed_mph
#        geojson_row['properties']['speed_m/s'] = speed_mps
#        #out.append(geojson_row)
#    # save file
#    with open(geojson_path_out, 'w') as f:
#        json.dump(out, f, ensure_ascii=False)

    return speed_mph_arr, speed_mph_set


###############################################################################
def update_geojson_dir_speed(geojson_dir_in, geojson_dir_out,
                             label_type='sn5', suffix='_speed', nmax=1000000,
                             verbose=True, super_verbose=False):
    '''Update geojson data to add inferred speed information for entire
    directory'''

    os.makedirs(geojson_dir_out, exist_ok=True)
    speed_mph_set = set()
    speed_mph_arr = []

    json_files = np.sort([j for j in os.listdir(geojson_dir_in)
                          if j.endswith('.geojson')])
    if super_verbose:
        print("json_files:", json_files)

    for i, json_file in enumerate(json_files):
        if i >= nmax:
            break
        if (i % 100) == 0:  # verbose:
            print(i, "/", len(json_files), json_file)
        geojson_path_in = os.path.join(geojson_dir_in, json_file)
        root, ext = json_file.split('.')
        geojson_path_out = os.path.join(geojson_dir_out,
                                        root + suffix + '.' + ext)
        sarr, sset = add_speed_to_geojson(geojson_path_in, geojson_path_out,
                                          label_type=label_type,
                                          verbose=verbose)
        speed_mph_arr.extend(sarr)
        speed_mph_set = speed_mph_set.union(sset)

    print("speed_mph_set:", sorted(list(speed_mph_set)))
    unique, counts = np.unique(speed_mph_arr, return_counts=True)
    print("speed_mph counts:")
    d = dict(zip(unique, counts))
    for k, v in d.items():
        print(" ", k, v)

    return



###############################################################################
def create_speed_gdf(image_path, geojson_path, mask_path_out_gray,
                     bin_conversion_func, mask_burn_val_key='burnValue',
                     buffer_distance_meters=2, buffer_roundness=1,
                     dissolve_by='inferred_speed_mps',  # 'speed_m/s', 
                     bin_conversion_key='speed_mph',
                     crs=None,  # {'init': 'epsg:4326'},
                     zero_frac_thresh=0.05,
                     verbose=False):

    '''Create buffer around geojson for speeds, use bin_conversion_func to
    assign values to the mask'''

    # get gdf_buffer
    try:
        inGDF = gpd.read_file(geojson_path)
    except:
        print("Can't read geojson:", geojson_path)
        # create emty mask
        h, w = skimage.io.imread(image_path).shape[:2]
        #h, w = cv2.imread(image_path, 0).shape[:2]
        mask_gray = np.zeros((h, w)).astype(np.uint8)
        skimage.io.imsave(mask_path_out_gray, mask_gray)
        # cv2.imwrite(mask_path_out, mask_gray)
        return []

    if len(inGDF) == 0:
        print("Empty mask for path:", geojson_path)
        # create emty mask
        h, w = skimage.io.imread(image_path).shape[:2]  
        # h, w = cv2.imread(image_path, 0).shape[:2]
        mask_gray = np.zeros((h, w)).astype(np.uint8)
        skimage.io.imsave(mask_path_out_gray, mask_gray)
        # cv2.imwrite(mask_path_out, mask_gray)
        return []
        
    # project
    projGDF = osmnx_funcs.project_gdf(inGDF, to_crs=crs)
    if verbose:
        print("inGDF.columns:", inGDF.columns)

    gdf_utm_buffer = projGDF.copy()
    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = gdf_utm_buffer.buffer(buffer_distance_meters,
                                                       buffer_roundness)
    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by=dissolve_by)
    gdf_utm_dissolve.crs = gdf_utm_buffer.crs
    gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)
    if verbose:
        print("gdf_buffer['geometry'].values[0]:",
              gdf_buffer['geometry'].values[0])

    # set burn values
    speed_arr = gdf_buffer[bin_conversion_key].values
    burnVals = [bin_conversion_func(s) for s in speed_arr]
    gdf_buffer[mask_burn_val_key] = burnVals

    # create mask
    gdf_to_array(gdf_buffer, image_path, mask_path_out_gray,
                            mask_burn_val_key=mask_burn_val_key,
                            verbose=verbose)

    # check to ensure no mask outside the image pixels (some images are
    # largely black)
    im_bgr = skimage.io.imread(image_path)
    # im_bgr = cv2.imread(image_path, 1)
    try: 
        im_gray = np.sum(im_bgr, axis=2)
        # check if im_gray is more than X percent black
        zero_frac = 1. - float(np.count_nonzero(im_gray)) / im_gray.size
        if zero_frac >= zero_frac_thresh:
            print("zero_frac:", zero_frac)
            print("create_speed_gdf(): checking to ensure masks are null where "
                  "image is null")
            # ensure the label doesn't extend beyond the image
            mask_gray = cv2.imread(mask_path_out_gray, 0)
            zero_locs = np.where(im_gray == 0)
            # set mask_gray to zero at location of zero_locs
            mask_gray[zero_locs] = 0
            # overwrite
            cv2.imwrite(mask_path_out_gray, mask_gray)
    except:
        # something is wrong with the image...
        pass 
        # h0, w0 = 100, 100
        # mask_gray = np.zeros((h0, w0)).astype(np.uint8)
        # skimage.io.imsave(mask_path_out_gray, mask_gray)

    return gdf_buffer


###############################################################################
def create_speed_gdf_v0(image_path, geojson_path, mask_path_out_gray,
                     bin_conversion_func, mask_burn_val_key='burnValue',
                     buffer_distance_meters=2, # buffer_roundness=1,
                     dissolve_by='inferred_speed_mps',  # 'speed_m/s', 
                     bin_conversion_key='speed_mph',
                     # crs={'init': 'epsg:4326'},
                     zero_frac_thresh=0.05,
                     verbose=False):

    '''Create buffer around geojson for speeds, use bin_conversion_func to
    assign values to the mask
    https://shapely.readthedocs.io/en/stable/manual.html#object.buffer
    '''

    # get gdf_buffer
    try:
        inGDF = gpd.read_file(geojson_path)
    except:
        print("Can't read geojson:", geojson_path)
        # create emty mask
        h, w = skimage.io.imread(image_path).shape[:2]
        #h, w = cv2.imread(image_path, 0).shape[:2]
        mask_gray = np.zeros((h, w)).astype(np.uint8)
        skimage.io.imsave(mask_path_out_gray, mask_gray)
        # cv2.imwrite(mask_path_out, mask_gray)
        return []

    if len(inGDF) == 0:
        print("Empty mask for path:", geojson_path)
        # create emty mask
        h, w = skimage.io.imread(image_path).shape[:2]  
        # h, w = cv2.imread(image_path, 0).shape[:2]
        mask_gray = np.zeros((h, w)).astype(np.uint8)
        skimage.io.imsave(mask_path_out_gray, mask_gray)
        # cv2.imwrite(mask_path_out, mask_gray)
        return []
        
    # project
    projGDF = osmnx_funcs.project_gdf(inGDF) #, to_crs=crs)
    if verbose:
        print("inGDF.columns:", inGDF.columns)

    gdf_utm_buffer = projGDF.copy()
    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = gdf_utm_buffer.buffer(buffer_distance_meters)
                                                     #  , buffer_roundness)
    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by=dissolve_by)
    gdf_utm_dissolve.crs = gdf_utm_buffer.crs
    gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)
    if verbose:
        print("gdf_buffer['geometry'].values[0]:",
              gdf_buffer['geometry'].values[0])

    # set burn values
    speed_arr = gdf_buffer[bin_conversion_key].values
    burnVals = [bin_conversion_func(s) for s in speed_arr]
    gdf_buffer[mask_burn_val_key] = burnVals

    # create mask
    gdf_to_array(gdf_buffer, image_path, mask_path_out_gray,
                            mask_burn_val_key=mask_burn_val_key,
                            verbose=verbose)

    # check to ensure no mask outside the image pixels (some images are
    # largely black)
    im_bgr = skimage.io.imread(image_path)
    # im_bgr = cv2.imread(image_path, 1)
    try: 
        im_gray = np.sum(im_bgr, axis=2)
        # check if im_gray is more than X percent black
        zero_frac = 1. - float(np.count_nonzero(im_gray)) / im_gray.size
        if zero_frac >= zero_frac_thresh:
            print("zero_frac:", zero_frac)
            print("create_speed_gdf(): checking to ensure masks are null where "
                  "image is null")
            # ensure the label doesn't extend beyond the image
            mask_gray = cv2.imread(mask_path_out_gray, 0)
            zero_locs = np.where(im_gray == 0)
            # set mask_gray to zero at location of zero_locs
            mask_gray[zero_locs] = 0
            # overwrite
            cv2.imwrite(mask_path_out_gray, mask_gray)
    except:
        # something is wrong with the image...
        pass 
        # h0, w0 = 100, 100
        # mask_gray = np.zeros((h0, w0)).astype(np.uint8)
        # skimage.io.imsave(mask_path_out_gray, mask_gray)

    return gdf_buffer


##############################################################################
def gdf_to_array(gdf, im_file, output_raster, burnValue=150,
                 mask_burn_val_key='', compress=True, NoData_value=0,
                 verbose=False):
    """
    Create buffer around geojson for desired geojson feature, save as mask

    Notes
    -----
    https://gis.stackexchange.com/questions/260736/how-to-burn-a-different-value-for-each-polygon-in-a-json-file-using-gdal-rasteri/260737


    Arguments
    ---------
    image_path : gdf
        Input geojson
    im_file : str
        Path to image file corresponding to gdf.
    output_raster : str
        Output path of saved mask (should end in .tif).
    burnValue : int
        Value to burn to mask. Superceded by mask_burn_val_key.
        Defaults to ``150``.
    mask_burn_val_key : str
        Column name in gdf to use for mask burning. Supercedes burnValue.
        Defaults to ``''`` (in which case burnValue is used).
    compress : bool
        Switch to compress output raster. Defaults to ``True``.
    NoData_value : int
        Value to assign array if no data exists. If this value is <0
        (e.g. -9999), a null value will show in the image. Defaults to ``0``.
    verbose : bool
        Switch to print relevant values.  Defaults to ``False``.

    Returns
    -------
    None
    """

    gdata = gdal.Open(im_file)

    # set target info
    if compress:
        target_ds = gdal.GetDriverByName('GTiff').Create(output_raster,
                                                         gdata.RasterXSize,
                                                         gdata.RasterYSize, 1,
                                                         gdal.GDT_Byte,
                                                         ['COMPRESS=LZW'])
    else:
        target_ds = gdal.GetDriverByName('GTiff').Create(output_raster,
                                                         gdata.RasterXSize,
                                                         gdata.RasterYSize, 1,
                                                         gdal.GDT_Byte)

    target_ds.SetGeoTransform(gdata.GetGeoTransform())
    if verbose:
        print("gdata.GetGeoTransform():", gdata.GetGeoTransform())

    

    # set raster info
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(gdata.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())
    if verbose:
        print ("target_ds:", target_ds)

    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)

    outdriver = ogr.GetDriverByName('MEMORY')
    outDataSource = outdriver.CreateDataSource('memData')
    tmp = outdriver.Open('memData', 1)
    outLayer = outDataSource.CreateLayer("states_extent", raster_srs,
                                         geom_type=ogr.wkbMultiPolygon)
    # burn
    burnField = "burn"
    idField = ogr.FieldDefn(burnField, ogr.OFTInteger)
    outLayer.CreateField(idField)
    featureDefn = outLayer.GetLayerDefn()
    for j, geomShape in enumerate(gdf['geometry'].values):
        if verbose:
            print (j, "geomshape:", geomShape)
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(ogr.CreateGeometryFromWkt(geomShape.wkt))
        if len(mask_burn_val_key) > 0:
            burnVal = int(gdf[mask_burn_val_key].values[j])
            if verbose:
                print("burnVal:", burnVal)
        else:
            burnVal = burnValue
        outFeature.SetField(burnField, burnVal)
        outLayer.CreateFeature(outFeature)
        # if verbose:
        #     print ("outFeature:", outFeature)
        outFeature = 0

    if len(mask_burn_val_key) > 0:
        gdal.RasterizeLayer(target_ds, [1], outLayer,
                            options=["ATTRIBUTE=%s" % burnField])
    else:
        gdal.RasterizeLayer(target_ds, [1], outLayer, burn_values=[burnVal])

    outLayer = 0
    outDatSource = 0
    tmp = 0
    return


##############################################################################
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--geojson_dir_in', default='', type=str,
                        help='Input directory of geojsons')
    parser.add_argument('--geojson_dir_out', default='', type=str,
                        help='Output directory of updated geojsons')
    parser.add_argument('--label_type', default='sn5', type=str,
                        help="sn3', 'sn5', or 'osm'")
    parser.add_argument('--suffix', default='_speed', type=str,
                        help='suffix for output')
    args = parser.parse_args()

    update_geojson_dir_speed(args.geojson_dir_in, args.geojson_dir_out,
                             label_type=args.label_type,
                             suffix=args.suffix,
                             verbose=True)

##############################################################################
if __name__ == "__main__":
    main()

    