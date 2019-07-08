import os
import scipy.misc
#from scipy.misc import imread
import skimage.io
import numpy as np

from dataset.abstract_image_type import AbstractImageType
from other_tools import apls_tools


class RawImageType(AbstractImageType):
    """
    image provider constructs image of type and then you can work with it
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha, num_channels):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        if num_channels == 3:
            self.im = scipy.misc.imread(os.path.join(self.paths['images'], self.fn), mode='RGB')
        else:
            self.im = apls_tools.load_multiband_im(os.path.join(self.paths['images'], self.fn),
                                                   method='gdal')
    def read_image(self):
        im = self.im[...,:-1] if self.has_alpha else self.im
        return self.finalyze(im)

    def read_mask(self, verbose=False):
        path = os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.fn))
        # AVE edit:
        mask_channels = skimage.io.imread(path)
        # skimage reads in (channels, h, w) for multi-channel
        # assume less than 20 channels
        #print ("mask_channels.shape:", mask_channels.shape)
        if mask_channels.shape[0] < 20: 
            #print ("mask_channels.shape:", mask_channels.shape)
            mask = np.moveaxis(mask_channels, 0, -1)
            #print ("mask.shape:", mask.shape)         
        else:
            mask = mask_channels
                    
        ## original version (mode='L' is a grayscale black and white image)
        #mask = scipy.misc.imread(path, mode='L')
        if verbose: 
            print ("raw_image.py mask.shape:", mask.shape)
            print ("raw_image.py np.unique mask", np.unique(mask))

        return self.finalyze(mask)

    def read_alpha(self):
        return self.finalyze(self.im[...,-1])

    def finalyze(self, data):
        return self.reflect_border(data)