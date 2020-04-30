import os
import sys
import time
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader
from tqdm import tqdm
from typing import Type

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.neural_dataset import TrainDataset, ValDataset
from .loss import dice_round, dice, focal_v0, focal, soft_dice_loss, \
    weight_reshape
from .callbacks import EarlyStopper, ModelSaver, TensorBoard, \
    CheckpointSaver, Callbacks, LRDropCheckpointSaver, ModelFreezer
from pytorch_zoo import unet

############
# need the following to avoid the following error:
#  TqdmSynchronisationWarning: Set changed size during iteration (see https://github.com/tqdm/tqdm/issues/481
from tqdm import tqdm
tqdm.monitor_interval = 0
############

torch.backends.cudnn.benchmark = True

models = {
    'resnet34': unet.Resnet34_upsample,
    'resnet50': unet.Resnet50_upsample,
    'resnet101': unet.Resnet101_upsample,
    #'resnet34_3channel': unet.Resnet34_upsample,
    #'resnet34_8channel': unet.Resnet34_upsample,
    'seresnet50': unet.SeResnet50_upsample,
    'seresnet101': unet.SeResnet101_upsample,
    'seresnet152': unet.SeResnet152_upsample,
    'seresnext50': unet.SeResnext50_32x4d_upsample,
    'seresnext101': unet.SeResnext101_32x4d_upsample,

}

optimizers = {
    'adam': optim.Adam,
    'rmsprop': optim.RMSprop,
    'sgd': optim.SGD
}

class Estimator:
    """
    incapsulates optimizer, model and make optimizer step
    """
    
    def __init__(self, model: torch.nn.Module, optimizer: Type[optim.Optimizer], save_path, config):
        self.model = nn.DataParallel(model).cuda()
        self.optimizer = optimizer(self.model.parameters(), lr=config.lr)
        self.start_epoch = 0
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.iter_size = config.iter_size

        self.lr_scheduler = None
        self.lr = config.lr
        self.config = config
        self.optimizer_type = optimizer

    def resume(self, checkpoint_name):
        try:
            checkpoint = torch.load(os.path.join(self.save_path, checkpoint_name))
        except FileNotFoundError:
            print("Attempt to resume failed, file not found")
            print ("  Missing file:", os.path.join(self.save_path, checkpoint_name))
            return False

        # # AVE edit:
        #   don't use previous epoch, instead start from scratch?
        #if checkpoint['epoch'] > 0:
        #    print ("train.py: Starting from epoch 0 instead of " \
        #           + "checkpoint['epoch'] =" + str(checkpoint['epoch']))
        self.start_epoch = checkpoint['epoch']

        model_dict = self.model.module.state_dict()
        pretrained_dict = checkpoint['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.module.load_state_dict(model_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        print("resumed from checkpoint {} on epoch: {}".format(os.path.join(self.save_path, checkpoint_name), self.start_epoch))
        return True

    def calculate_loss_single_channel(self, output, target, meter, training, 
                                      iter_size, weight_channel=None):
        # apply weights and reshapes if needed
        if weight_channel:
            output, target = weight_reshape(output, target,
                                            weight_channel=weight_channel,
                                            min_weight_val=0.16)

        bce = F.binary_cross_entropy_with_logits(output, target)        
        if 'ce' in self.config.loss.keys():
            pass
        else:
            output = F.sigmoid(output)
        
        #ce = F.cross_entropy(output, target)
        #ce = F.cross_entropy(output.long(), target)
        d = dice(output, target)
        dice_l = 1 - d
        dice_soft_l = soft_dice_loss(output, target)
        focal_l = focal(output, target)

        smooth_l1_mult = 100
        smooth_l1_l = F.smooth_l1_loss(output, target) * smooth_l1_mult
        
        mse_mult = 10
        mse_l = F.mse_loss(output, target) * mse_mult
        
        # jacc = jaccard(output, target)
        # dice_r = dice_round(output, target)
        # jacc_r = jaccard_round(output, target)

        # custom loss function
        # AVE edit
        if 'focal_v0' in self.config.loss.keys():
            loss = (self.config.loss['focal_v0'] * focal_v0(output, target) + self.config.loss['dice'] * (1 - d) ) / iter_size
        elif 'bce' in self.config.loss.keys():
            loss = (self.config.loss['bce'] * bce + self.config.loss['dice'] * (1 - d)) / iter_size
        elif 'focal' in self.config.loss.keys():
            focal_l = focal(output, target)
            loss = (self.config.loss['focal'] * focal_l + self.config.loss['soft_dice'] * dice_soft_l) / iter_size
        elif 'smooth_l1' in self.config.loss.keys():
            loss = (self.config.loss['smooth_l1'] * smooth_l1_l + self.config.loss['dice'] * (1 - d)) / iter_size
        elif 'mse' in self.config.loss.keys():
            loss = (self.config.loss['mse'] * mse_l + self.config.loss['dice'] * (1 - d)) / iter_size
        #elif 'ce' in self.config.loss.keys():
        #    loss = (self.config.loss['ce'] * ce + self.config.loss['soft_dice'] * dice_soft_l) / iter_size
            
        if training:
            loss.backward()

        meter['tot_loss'] += loss.data.cpu().numpy()
        # meter['tot_loss'] += loss.data.cpu().numpy()[0]
        
        #meter['bce'] += bce.data.cpu().numpy()[0] / iter_size
        
        meter['focal'] += focal_l.data.cpu().numpy() / iter_size
        # meter['focal'] += focal_l.data.cpu().numpy()[0] / iter_size

        #meter['ce'] += ce.data.cpu().numpy()[0] / iter_size
        #meter['dice_round'] += dice_r.data.cpu().numpy()[0] / iter_size
        # meter['jr'] += jacc_r.data.cpu().numpy()[0] / iter_size
        # meter['jacc'] += jacc.data.cpu().numpy()[0] / iter_size
        #meter['dice'] += d.data.cpu().numpy()[0] / iter_size
        
        meter['dice_loss'] += dice_l.data.cpu().numpy() / iter_size
        # meter['dice_loss'] += dice_l.data.cpu().numpy()[0] / iter_size

        # meter['smooth_l1'] += smooth_l1_l.data.cpu().numpy()[0] / iter_size

        meter['mse'] += mse_l.data.cpu().numpy() / iter_size
        # meter['mse'] += mse_l.data.cpu().numpy()[0] / iter_size
        
        return meter

    def make_step_itersize(self, images, ytrues, training, verbose=False):
        iter_size = self.iter_size
        
        if verbose:
            print("images.shape:", images.shape)
            print("ytrues.shape:", ytrues.shape)
        
        if training:
            self.optimizer.zero_grad()

        inputs = images.chunk(iter_size)
        targets = ytrues.chunk(iter_size)
        
        if verbose:
            print("len inputs", len(inputs))
            print("len shape:", len(targets))

        meter = defaultdict(float)
        for input, target in zip(inputs, targets):
            input = torch.autograd.Variable(input.cuda(async=True), volatile=not training)
            target = torch.autograd.Variable(target.cuda(async=True), volatile=not training)
            if verbose:
                print("input.shape, target.shape:", input.shape, target.shape)
            output = self.model(input)
            meter = self.calculate_loss_single_channel(output, target, meter, training, iter_size)

        if training:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.)
            self.optimizer.step()
        return meter, None#torch.cat(outputs, dim=0)

class MetricsCollection:
    def __init__(self):
        self.stop_training = False
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class PytorchTrain:
    """
    fit, run one epoch, make step
    """
    def __init__(self, estimator: Estimator, fold, callbacks=None, hard_negative_miner=None):
        self.fold = fold
        self.estimator = estimator
        #print ("pytorch_utils.train.py PyTorchTrain test0")

        self.devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
        #print ("pytorch_utils.train.py os.name", os.name)
        #print ("pytorch_utils.train.py self.devices", self.devices)
        if os.name == 'nt':
            self.devices = ','.join(str(d + 5) for d in map(int, self.devices.split(',')))

        self.hard_negative_miner = hard_negative_miner
        self.metrics_collection = MetricsCollection()

        self.estimator.resume("fold" + str(fold) + "_checkpoint.pth")

        self.callbacks = Callbacks(callbacks)
        self.callbacks.set_trainer(self)
        #print ("pytorch_utils.train.py PyTorchTrain test1")


    def _run_one_epoch(self, epoch, loader, training=True, verbose=False):
        avg_meter = defaultdict(float)
        
        #print ("Sometimes a problem in pytorch_utils.train.py _run_one_epoch()" \
        #       + " this is caused by image_cropper if target_cols is too large")
        if verbose:
            print("epoch:", epoch)
            print ("len(loader):", len(loader))
        #print ("loader:", loader)
            
        pbar = tqdm(enumerate(loader), total=len(loader), desc="Fold {}; Epoch {}{}".format(self.fold, epoch, ' eval' if not training else ""), ncols=0)
        for i, data in pbar:
            self.callbacks.on_batch_begin(i)

            meter, ypreds = self._make_step(data, training)
            for k, val in meter.items():
                avg_meter[k] += val

            if training:
                if self.hard_negative_miner is not None:
                    self.hard_negative_miner.update_cache(meter, data)
                    if self.hard_negative_miner.need_iter():
                        self._make_step(self.hard_negative_miner.cache, training)
                        self.hard_negative_miner.invalidate_cache()
            
            #print ("pytorch_utils.train.py PyTorchTrain test2")
            #print ("avg_meter.items():", avg_meter.items())
            pbar.set_postfix(**{k: "{:.5f}".format(v / (i + 1)) for k, v in avg_meter.items()})

            self.callbacks.on_batch_end(i)
        return {k: v / len(loader) for k, v in avg_meter.items()}

    def _make_step(self, data, training, verbose=False):
        images = data['image']
        ytrues = data['mask']
        if verbose:
            print("images shapes:", [z.shape for z in images])
            print("ytrues shapes:", [z.shape for z in ytrues])
        meter, ypreds = self.estimator.make_step_itersize(images, ytrues, training)

        return meter, ypreds

    def fit(self, train_loader, val_loader, nb_epoch, logger=None):
        self.callbacks.on_train_begin()

        t0 = time.time()
        for epoch in range(self.estimator.start_epoch, nb_epoch):
            t1 = time.time()
            self.callbacks.on_epoch_begin(epoch)

            if self.estimator.lr_scheduler is not None:
                self.estimator.lr_scheduler.step(epoch)

            self.estimator.model.train()
            # print("pytorch_utils.train.py.fit() checkpoint0")
            self.metrics_collection.train_metrics = self._run_one_epoch(epoch, train_loader, training=True)
            self.estimator.model.eval()
            # print("pytorch_utils.train.py.fit() checkpoint1")
            self.metrics_collection.val_metrics = self._run_one_epoch(epoch, val_loader, training=False)
            # print("pytorch_utils.train.py.fit() checkpoint2")
            t2 = time.time()
            #logger.info("folds_file_loc: {}".format(folds_file_loc))
            dt = np.round( (t2 - t1)/60.0, 1)
            dt_tot = np.round( (t2 - t0)/60.0, 1)
            if logger:
                logger.info("train epoch {}, time elapsed (minutes): {}".format(epoch, dt))
                logger.info("  train epoch {}, train loss: {} ".format(epoch, self.metrics_collection.train_metrics))
                logger.info("  train epoch {}, val loss: {} ".format(epoch, self.metrics_collection.val_metrics))
                logger.info("  train epoch {}, total time elapsed (minutes): {}".format(epoch, dt_tot))
            print("epoch", epoch, "dt:", dt, "minutes")
            print("Total time elapsed:", dt_tot, "minutes")
            self.callbacks.on_epoch_end(epoch)
            if self.metrics_collection.stop_training:
                logger.info("  callback stop training issued...")
                logger.info("  callbacks.on_epoch_end(epoch) ".format(self.callbacks.on_epoch_end(epoch)))
                break

        self.callbacks.on_train_end()


def train(ds, fold, train_idx, val_idx, config, save_path, log_path, 
          val_ds=None, num_workers=0, transforms=None, val_transforms=None,
          logger=None):
    #os.makedirs(os.path.join(config.results_dir, 'weights'), exist_ok=True)
    #os.makedirs(os.path.join(config.results_dir, 'logs'), exist_ok=True)
    #save_path = os.path.join(config.results_dir, 'weights', config.folder)
    model = models[config.network](num_classes=config.num_classes, num_channels=config.num_channels)
    print("model:", model)
    if logger:
        logger.info("pytorch_utils train.py config.num_channels: {}".format(config.num_channels))
        logger.info("pytorch_utils train.py function train(),  model: {}".format(model))
    else:
        print("pytorch_utils train.py config.num_channels:", config.num_channels)
        print ("pytorch_utils train.py function train(),  model:", model)
    estimator = Estimator(model, optimizers[config.optimizer], save_path, config=config)
    #print("pytorch_utils train.py estimator:", estimator)

    estimator.lr_scheduler = MultiStepLR(estimator.optimizer, config.lr_steps, gamma=config.lr_gamma)
    callbacks = [
        ModelSaver(1, ("fold"+str(fold)+"_best.pth"), best_only=True),
        ModelSaver(1, ("fold"+str(fold)+"_last.pth"), best_only=False),
        CheckpointSaver(1, ("fold"+str(fold)+"_checkpoint.pth")),
        # LRDropCheckpointSaver(("fold"+str(fold)+"_checkpoint_e{epoch}.pth")),
        # ModelFreezer(),
        TensorBoard(os.path.join(log_path, config.save_weights_dir, 'fold{}'.format(fold))),
        #TensorBoard(os.path.join(config.results_dir, 'logs', config.save_weights_name, 'fold{}'.format(fold)))
        # AVE edit:
        EarlyStopper(config.early_stopper_patience)
    ]

    #print ("pytorch_utils.train.py test0")
    trainer = PytorchTrain(estimator,
                           fold=fold,
                           callbacks=callbacks,
                           hard_negative_miner=None)
    #print ("pytorch_utils.train.py test1")

    #z = TrainDataset(ds, train_idx, config, transforms=transforms)
    #print ("TrainDataSet:", z)
    #print ("len TrainDataSet:", len(z))
    print("pytorch_utils.train.py len train_idx", len(train_idx))
    train_loader = PytorchDataLoader(TrainDataset(ds, train_idx, config, transforms=transforms),
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=num_workers,
                                     pin_memory=True)
    print("pytorch_utils.train.py len train_loader", len(train_loader))
    print("  (len train_loader is num images * 8 / batch_size)")
    val_loader = PytorchDataLoader(ValDataset(val_ds if val_ds is not None else ds, val_idx, config, transforms=val_transforms),
                                   batch_size=config.batch_size if not config.ignore_target_size else 1,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=num_workers,
                                   pin_memory=True)
    print("pytorch_utils.train.py len val_loader:", len(val_loader))
    print("Run trainer.fit in pytorch_utils.train.py...")
    trainer.fit(train_loader, val_loader, config.nb_epoch, logger=logger)
