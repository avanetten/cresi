import os
import sys
import cv2
import warnings
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import skimage.io

from .eval import Evaluator

# import relative paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.save_array_gdal import CreateMultiBandGeoTiff

# skimage gives really annoying warnings
warnings.filterwarnings("ignore")
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")


class FullImageEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, predicted, model, data, prefix="", verbose=False):
        names = data['image_name']
        if verbose:
            print("concrete_eval.py.process_batch()  predicted.shape;", predicted.shape,)
        for i in range(len(names)):
            self.on_image_constructed(names[i], predicted[i,...], prefix)

    def save(self, name, prediction, prefix="", #save_im_gdal_format=True,
             verbose=False):
        # AVE edit
        save_im_gdal_format = self.save_im_gdal_format
        if verbose:
            print ("concrete_eval.py: prediction.shape:", prediction.shape)
            print ("np.unique prediction:", np.unique(prediction))
        if len(prediction.shape) == 2:
            cv2.imwrite(os.path.join(self.save_dir, prefix + name), (prediction * 255).astype(np.uint8))
        else:

            # skimage reads in (channels, h, w) for multi-channel
            # assume less than 20 channels
            #print ("mask_channels.shape:", mask_channels.shape)
            if prediction.shape[0] > 20: 
                #print ("mask_channels.shape:", mask_channels.shape)
                mask = np.moveaxis(prediction, -1, 0)
                #print ("mask.shape:", mask.shape)         
            else:
                mask = prediction
            if verbose:
                print ("concrete_eval.py: mask.shape:", mask.shape)
            
            # save with skimage
            outfile_sk = os.path.join(self.save_dir, prefix + name)
            if verbose:
                print ("name:", name)
                print ("mask.shape:", mask.shape)
                print ("prediction.shape:", prediction.shape)
                print ("outfile_sk:", outfile_sk)
            skimage.io.imsave(outfile_sk, (mask * 255).astype(np.uint8), 
                              compress=1)
            
            # also save with gdal?
            if save_im_gdal_format:
                save_dir_gdal = os.path.join(self.save_dir + '_gdal')
                #print ("save_dir_gdal:", save_dir_gdal)
                os.makedirs(save_dir_gdal, exist_ok=True)
                CreateMultiBandGeoTiff(os.path.join(save_dir_gdal, prefix + name), (mask * 255).astype(np.uint8))
              

class CropEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_mask = None
        self.current_prediction = None
        self.current_image_name = None

    def process_batch(self, predicted, model, data, prefix=""):
        names = data['image_name']
        config = self.config
        batch_geometry = self.parse_geometry(data['geometry'])
        for i in range(len(names)):
            name = names[i]
            geometry = batch_geometry[i]
            sx, sy = geometry['sx'], geometry['sy']
            pred = self.cut_border(np.squeeze(predicted[i,...]))
            if name != self.current_image_name:
                if self.current_image_name is None:
                    self.current_image_name = name
                else:
                    self.on_image_constructed(self.current_image_name, self.current_prediction / self.current_mask, prefix=prefix)
                self.construct_big_image(geometry)
            self.current_prediction[sy + self.border:sy + config.target_rows - self.border, sx + self.border:sx + config.target_cols - self.border] += pred
            self.current_mask[sy+self.border:sy + config.target_rows - self.border, sx + self.border:sx + config.target_cols - self.border] += 1
            self.current_image_name = name

    def parse_geometry(self, batch_geometry):
        rows = batch_geometry['rows'].numpy()
        cols = batch_geometry['cols'].numpy()
        sx = batch_geometry['sx'].numpy()
        sy = batch_geometry['sy'].numpy()
        geometries = []
        for idx in range(rows.shape[0]):
            geometry = {'rows': rows[idx],
                        'cols': cols[idx],
                        'sx': sx[idx],
                        'sy': sy[idx]}
            geometries.append(geometry)
        return geometries

    def construct_big_image(self, geometry):
        self.current_mask = np.zeros((geometry['rows'], geometry['cols']), np.uint8)
        self.current_prediction = np.zeros((geometry['rows'], geometry['cols']), np.float32)

    def save(self, name, prediction, prefix="", #save_im_gdal_format=True,
             verbose=False):
        # AVE edit
        save_im_gdal_format = self.save_im_gdal_format
        if verbose:
            print ("concrete_eval.py: prediction.shape:", prediction.shape)
            print ("np.unique prediction:", np.unique(prediction))
        if len(prediction.shape) == 2:
            cv2.imwrite(os.path.join(self.save_dir, prefix + name), (prediction * 255).astype(np.uint8))
        else:

            # skimage reads in (channels, h, w) for multi-channel
            # assume less than 20 channels
            #print ("mask_channels.shape:", mask_channels.shape)
            if prediction.shape[0] > 20: 
                #print ("mask_channels.shape:", mask_channels.shape)
                mask = np.moveaxis(prediction, -1, 0)
                #print ("mask.shape:", mask.shape)         
            else:
                mask = prediction
            if verbose:
                print ("concrete_eval.py: mask.shape:", mask.shape)
            
            # save with skimage
            outfile_sk = os.path.join(self.save_dir, prefix + name)
            if verbose:
                print ("name:", name)
                print ("mask.shape:", mask.shape)
                print ("prediction.shape:", prediction.shape)
                print ("outfile_sk:", outfile_sk)
            skimage.io.imsave(outfile_sk, (mask * 255).astype(np.uint8), 
                              compress=1)
            
            # also save with gdal?
            if save_im_gdal_format:
                save_dir_gdal = os.path.join(self.save_dir + '_gdal')
                #print ("save_dir_gdal:", save_dir_gdal)
                os.makedirs(save_dir_gdal, exist_ok=True)
                CreateMultiBandGeoTiff(os.path.join(save_dir_gdal, prefix + name), (mask * 255).astype(np.uint8))
              