#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 09:28:17 2018

@author: avanetten

"""

from __future__ import print_function
from statsmodels.stats.weightstats import DescrStatsW

###############################################################################
def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    
    weighted_stats = DescrStatsW(values, weights=weights, ddof=0)

    mean = weighted_stats.mean      # weighted mean of data (equivalent to np.average(array, weights=weights))
    std = weighted_stats.std       # standard deviation with default degrees of freedom correction
    var = weighted_stats.var              # variance with default degrees of freedom correction

    return (mean, std, var)