#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:07:20 2019

@author: avanetten

https://discuss.pytorch.org/t/repeat-a-tensor-and-concat-them-along-certain-dimension/9637
https://pytorch.org/docs/stable/tensors.html
https://github.com/pytorch/pytorch/issues/13773
https://pytorch.org/docs/stable/_modules/torch/nn/functional.html
https://discuss.pytorch.org/t/expand-a-2d-tensor-to-3d-tensor/9614
https://stackoverflow.com/questions/44524901/how-to-do-product-of-matrices-in-pytorch
https://www.aiworkbox.com/lessons/clip-pytorch-tensor-values-to-a-range
https://pytorch.org/docs/master/torch.html?#torch.clamp
https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
https://stackoverflow.com/questions/48915810/pytorch-contiguous

"""

import torch
import numpy as np
# import matplotlib.pyplot as plt


def weight_reshape_torch(preds, trues, weight_channel=-1, min_weight_val=0.16):
    """
    Weight the prediction by the desired channel in trues.  Assume tensor has
    shape: (batch_size, channels, h, w)
    Also assume weight_channel = -1
    Clip weights by min_weight_val.

    Test:
        preds = torch.randn(10,8,6,4)
        trues = torch.randn(10,8,6,4)
        weights_channel = 3*torch.ones(weights_channel.shape)

    Return updated trues without the weight channel, and updated preds
    withouth the weight channel and multiplied by weights
    """

    # batch_size = trues.size()[0]
    trues_vals = trues[:, 0:weight_channel, :, :]
    preds_vals = preds[:, 0:weight_channel, :, :]

    weights_channel = trues[:, weight_channel, :, :]
    # expand weights to same size as trues_vals (not sure how!)
    # weights = weights_channel.expand(trues_vals.shape)

    # simple, slow method, loop over the stack
    # element wise multiply weights by preds_vals
    for channel in range(preds_vals.shape[1]):
        x = preds_vals[:, channel, :, :]
        out = torch.mul(x, weights_channel)
        preds_vals[:, channel, :, :] = out

    return preds_vals, trues_vals


def weight_reshape_np(preds, trues, weight_channel=-1, min_weight_val=0.16,
                      verbose=True):
    """
    Weight the prediction by the desired channel in trues.  Assume tensor has
    shape: (channels, h, w)
    Also assume weight_channel = -1
    Clip weights by min_weight_val.

    Test:
        preds = torch.randn(10,8,6,4)
        trues = torch.randn(10,8,6,4)
        weights_channel = 3*torch.ones(weights_channel.shape)

    Return updated trues without the weight channel, and updated preds
    withouth the weight channel and multiplied by weights
    """

    # batch_size = trues.size()[0]
    weights = trues[weight_channel, :, :].astype(float)

    # strip out final channel containing weights
    trues_vals = trues[0:weight_channel, :, :]
    # preds_vals = preds[0:weight_channel, :, :]

    if verbose:
        print("preds_vals.shape:", preds.shape)
        print("trues_vals.shape:", trues_vals.shape)
        print("np.max(weights):", np.max(weights))

    # normalize
    if np.max(weights) <= 1:
        pass
    elif 1 < np.max(weights) < 255.:
        weights *= 1./255
    else:
        print("Unknown weight scale...")
        return
    # clip
    weights = np.clip(weights, min_weight_val, np.max(weights)).astype(float)

    # expand weights to same size as trues_vals (not sure how!)
    # weights = weights_channel.expand(trues_vals.shape)
    weights_expand = np.ones((preds.shape))
    for channel in range(weights_expand.shape[0]):
        weights_expand[channel, :, :] = weights

    preds_rw = np.multiply(preds, weights_expand)

#    # simple, slow method, loop over the stack
#    # element wise multiply weights by preds_vals
#    for channel in range(preds_vals.shape[0]):
#        x = preds_vals[channel, :, :]
#        # if verbose:
#        #     print(" x.shape:", x.shape)
#        out = np.multiply(x.astype(float), weights)
#        preds_vals[channel, :, :] = out
#        # plt.imshow(out)

    return preds_rw, trues_vals, weights
