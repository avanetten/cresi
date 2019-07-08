#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:42:15 2019

@author: avanetten
"""

import logging

###############################################################################
def make_logger(log_file, logger_name='log'):
    # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO,
    #logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                        #format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_file,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-8s: %(levelname)-8s %(message)s')
    #formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger0 = logging.getLogger(logger_name)
    return console, logger0
      
