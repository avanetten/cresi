import numpy as np
import torch
import torch.nn.functional as F
eps = 1


def weight_reshape(preds, trues, weight_channel=-1, min_weight_val=0.16):
    """
    UNTESTED
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


def dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return dice(preds, trues, is_average=is_average)


def dice(preds, trues, weight=None, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    score = scores.sum()
    if is_average:
        score /= num
    return torch.clamp(score, 0., 1.)


def focal_v0(preds, trues, alpha=1, gamma=2, reduce=True, logits=True):
    '''https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938'''
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(preds, trues)#, reduce=False)
    else:
        BCE_loss = F.binary_cross_entropy(preds, trues)#, reduce=False)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss


##########
def soft_dice_loss(outputs, targets, per_image=False):
    '''
    From cannab sn4
    '''
    
    batch_size = outputs.size()
    # batch_size = outputs.size()[0]
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss


def dice_cannab_v0(im1, im2, empty_score=1.0):
    """
    From cannab sn4
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def focal(outputs, targets, gamma=2,  ignore_index=255):
    '''From cannab sn4'''
    outputs = outputs.contiguous()
    targets = targets.contiguous()
    eps = 1e-8
    non_ignored = targets.view(-1) != ignore_index
    targets = targets.view(-1)[non_ignored].float()
    outputs = outputs.contiguous().view(-1)[non_ignored]
    outputs = torch.clamp(outputs, eps, 1. - eps)
    targets = torch.clamp(targets, eps, 1. - eps)
    pt = (1 - targets) * (1 - outputs) + targets * outputs
    return (-(1. - pt) ** gamma * torch.log(pt)).mean()

