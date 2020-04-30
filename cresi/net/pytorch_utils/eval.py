import os
import sys
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.backends.cudnn.benchmark = True
import tqdm
from multiprocessing.pool import Pool
from torch import multiprocessing

from torch.serialization import SourceChangeWarning
import warnings
import torchsummary
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader

# import relative paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset.neural_dataset import SequentialDataset

# global variables for cpu_func()
MOD = 0
FLIPS = 0
BORDER = 0
PREFIX = 0
SAVE_DIR = 0


class flip:
    FLIP_NONE=0
    FLIP_LR=1
    FLIP_FULL=2


def flip_tensor_lr(batch):
    columns = batch.data.size()[-1]
    if torch.cuda.is_available():
        index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(columns)))).cuda())
    else:
        index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(columns)))))
        
    return batch.index_select(3, index)


def flip_tensor_ud(batch):
    rows = batch.data.size()[-2]
    if torch.cuda.is_available():
        index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(rows)))).cuda())
    else:
        index = torch.autograd.Variable(torch.LongTensor(list(reversed(range(rows)))))

    return batch.index_select(2, index)


def to_numpy(batch):
    return np.moveaxis(batch.data.cpu().numpy(), 1, -1)


def predict(model, batch, flips=flip.FLIP_NONE, verbose=False):
    
    if verbose:
        print("  eval.py - predict() - executing...")

    #print ("run eval.predict()...")
    # predict with tta on gpu
    pred1 = F.sigmoid(model(batch))
    # with torch.no_grad():
    #    pred1 = F.sigmoid(model(batch))

    if verbose:
        print("  eval.py - predict() - batch.shape:", batch.shape)
        print("  eval.py - predict() - pred1.shape:", pred1.shape)

    if flips > flip.FLIP_NONE:
        pred2 = flip_tensor_lr(model(flip_tensor_lr(batch)))
        masks = [pred1, pred2]
        if flips > flip.FLIP_LR:
            pred3 = flip_tensor_ud(model(flip_tensor_ud(batch)))
            pred4 = flip_tensor_ud(flip_tensor_lr(model(flip_tensor_ud(flip_tensor_lr(batch)))))
            masks.extend([pred3, pred4])
        masks = list(map(F.sigmoid, masks))
        new_mask = torch.mean(torch.stack(masks, 0), 0)
        return to_numpy(new_mask)
    return to_numpy(pred1)


#def read_model(config, fold):
def read_model(path_model_weights, fold, n_gpus=4):
    print ("Running eval.read_model()...")
    # model = nn.DataParallel(torch.load(os.path.join('..', 'weights', project, 'fold{}_best.pth'.format(fold))))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SourceChangeWarning)
        
        if torch.cuda.is_available():
            print ("load model with cuda")
            model = torch.load(os.path.join(path_model_weights, 'fold{}_best.pth'.format(fold))).cuda()
            #model = torch.load(os.path.join(config.path_model_weights, 'fold{}_best.pth'.format(fold)))
            # multple gpus
            try:
                model = nn.DataParallel(model, device_ids=[0,1,2,3], dim=0)
                print("multi-gpu")
            except:
                pass
        
        else:
            print("load model with cpu")
            # torch 0.3
            model = torch.load(os.path.join(path_model_weights, 'fold{}_best.pth'.format(fold)),  map_location=lambda storage, loc: storage)
            #model = torch.load(os.path.join(path_model_weights, 'fold{}_best.pth'.format(fold)),  map_location=lambda storage, location: 'cpu')
            ## torch 0.4
            #model = torch.load(os.path.join(path_model_weights, 'fold{}_best.pth'.format(fold)), map_location='cpu')
    
        #model = torch.load(os.path.join(config.results_dir, 'weights', config.folder, 'fold{}_best.pth'.format(fold)))
        model.eval()
        print ("  model:", model)
        print ("  model sucessfully loaded")
        return model
        

class Evaluator:
    """
    base class for evaluators
    """
    def __init__(self, config, ds, save_dir='', test=False, flips=0, 
                 num_workers=0, border=12, val_transforms=None,
                 weight_dir='', save_im_gdal_format=True,
                 #save_im_skimage=False
                 ):
        self.config = config
        self.ds = ds
        self.test = test
        self.flips = flips
        self.num_workers = num_workers

        self.current_prediction = None
        self.need_to_save = False
        self.border = border
        #self.folder = config.folder

        self.save_dir = save_dir
        self.weight_dir = weight_dir
        
        self.save_im_gdal_format = save_im_gdal_format

        self.val_transforms = val_transforms
        os.makedirs(self.save_dir, exist_ok=True)
        
        
    def predict(self, fold, val_indexes, weight_dir,
                verbose=False):
        global MOD
        global FLIPS
        global BORDER
        global PREFIX
        global SAVE_DIR
        n_threads_cpu = 12

        print ("run eval.Evaluator.predict()...")
        prefix = ('fold' + str(fold) + "_") if (self.test and fold is not None) else ""
        print ("prefix:", prefix)
        print ("Creating datasets within pytorch_utils/eval.py()...")
        if not torch.cuda.is_available():
            self.num_workers = n_threads_cpu

        val_dataset = SequentialDataset(self.ds, val_indexes, 
                                        stage='test', config=self.config, 
                                        transforms=self.val_transforms)
        val_dl = PytorchDataLoader(val_dataset, 
                                   batch_size=self.config.predict_batch_size,
                                   num_workers=self.num_workers, 
                                   drop_last=False)
        print("len val_dl:", len(val_dl))
        print("self.num_workers", self.num_workers)
        #print ("weights_dir:", self.weights_dir)
        model = read_model(weight_dir, fold)
        
        # set global variables
        FLIPS = self.flips
        BORDER = self.border
        SAVE_DIR = self.save_dir
        PREFIX = prefix
        MOD = model
        
        pbar = tqdm.tqdm(val_dl, total=len(val_dl))
        if torch.cuda.is_available():
            for data in pbar:
                samples = torch.autograd.Variable(data['image'], volatile=True).cuda()                    
                predicted = predict(model, samples, flips=self.flips)
                if verbose:
                    print("  eval.py -  - Evaluator - predict() - len samples:", len(samples))
                    print("  eval.py - Evaluator - predict()- samples.shape:", samples.shape)
                    print("  eval.py - Evaluator - predict() - predicted.shape:", predicted.shape)
                    print("  eval.py - Evaluator - predict() - data['image'].shape:", data['image'].shape)
                self.process_batch(predicted, model, data, prefix=prefix)
        else:
            for data in pbar:
                samples = torch.autograd.Variable(data['image'], volatile=True)
                predicted = predict(model, samples, flips=self.flips)
                if verbose:
                    print("  eval.py -  - Evaluator - predict() - len samples:", len(samples))
                    print("  eval.py - Evaluator - predict()- samples.shape:", samples.shape)
                    print("  eval.py - Evaluator - predict() - predicted.shape:", predicted.shape)
                    print("  eval.py - Evaluator - predict() - data['image'].shape:", data['image'].shape)
                self.process_batch(predicted, model, data, prefix=prefix)
        self.post_predict_action(prefix=prefix)


    def cut_border(self, image):
        if image is None:
            return None
        return image if not self.border else image[self.border:-self.border, self.border:-self.border, ...]

    def on_image_constructed(self, name, prediction, prefix=""):
        prediction = self.cut_border(prediction)
        prediction = np.squeeze(prediction)
        self.save(name, prediction, prefix=prefix)

    def save(self, name, prediction, prefix=""):
        raise NotImplementedError

    def process_batch(self, predicted, model, data, prefix=""):
        raise NotImplementedError

    def post_predict_action(self, prefix):
        pass
