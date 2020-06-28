# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn import Parameter
from typing import Tuple
from .utils import _tranpose_and_gather_feat


def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    """ Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    """
    pos_inds = gt.eq(1).float()  # ground truth为1的表示正样本像素点
    neg_inds = gt.lt(1).float()  # ground truth小于1的表示负样本点

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss(nn.Module):
    """
    nn.Module warpper for focal loss
    """

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


# 自定义用于多类分类的Focal loss
class McFocalLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 device,
                 use_alpha=False,
                 alpha=None,
                 gamma=2.0,
                 size_average=True):
        """
        :param num_classes:
        :param device:
        :param alpha:
        :param gamma:
        :param use_alpha:
        :param size_average:
        """
        super(McFocalLoss, self).__init__()

        self.num_classes = num_classes
        self.dev = device
        self.alpha = alpha
        self.gamma = gamma

        if use_alpha:
            self.alpha = torch.tensor(alpha).to(self.dev)

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1, self.num_classes))
        prob = prob.clamp(min=0.0001, max=1.0)

        target_ = torch.zeros(target.size(0), self.num_classes).to(self.dev)
        target_.scatter_(1, target.view(-1, 1).long(), 1.0)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1 - prob, self.gamma).double() \
                         * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1 - prob, self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


# Arc loss的FC layer用于细粒度分类或Re-ID
class ArcMarginFc(nn.Module):
    r"""
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self,
                 in_features,
                 out_features,
                 device,
                 s=30.0,
                 m=0.50,
                 easy_margin=False):
        """
        ArcMargin
        :type in_features: int
        :type out_features: int
        :param in_features:
        :param out_features:
        :param s:
        :param m:
        :param easy_margin:
        """
        super(ArcMarginFc, self).__init__()

        self.device = device
        self.in_dim = in_features
        self.out_dim = out_features
        print('=> in dim: %d, out dim: %d' % (self.in_dim, self.out_dim))

        self.s = s
        self.m = m

        # 根据输入输出dim确定初始化权重
        self.weight = Parameter(torch.FloatTensor(self.out_dim, self.in_dim))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # L2 normalize and calculate cosine
        cosine = F.linear(F.normalize(input, p=2), F.normalize(self.weight, p=2))

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # phi: cos(θ+m)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.device)  # device='cuda'
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # you can use torch.where if your torch.__version__ >= 0.4
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(output)

        return output


# -----Circle loss
def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class RegLoss(nn.Module):
    '''Regression loss for an output tensor
    Smooth L1 loss
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)

        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        """
        :param output:
        :param mask:
        :param ind:
        :param target:
        :return:
        """
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()

        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)  # 计算平均值
        return loss


class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        """
        :param output:
        :param mask:
        :param ind:
        :param target:
        :return:
        """
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _tranpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss


def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')


def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss
