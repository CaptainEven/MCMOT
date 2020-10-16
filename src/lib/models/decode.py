from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat


def _max_pool(heat, kernel=3):
    """
    NCHW
    do max pooling operation
    """
    # print("heat.shape: ", heat.shape)  # default: torch.Size([1, 1, 152, 272])

    pad = (kernel - 1) // 2

    h_max = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    # print("h_max.shape: ", h_max.shape)  # default: torch.Size([1, 1, 152, 272])

    keep = (h_max == heat).float()  # 将boolean类型的Tensor转换成Float类型的Tensor
    # print("keep.shape: ", keep.shape, "keep:\n", keep)
    return heat * keep


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def _topk(heatmap, K=40, num_classes=1):
    """
    scores=heatmap by default
    """
    N, C, H, W = heatmap.size()

    # 2d feature map -> 1d feature map
    topk_scores, topk_inds = torch.topk(heatmap.view(N, C, -1), K)

    topk_inds = topk_inds % (H * W)  # 这一步貌似没必要...
    # print("topk_inds.shape: ", topk_inds.shape)  # 1×1×128

    topk_ys = (topk_inds / W).int().float()
    topk_xs = (topk_inds % W).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(N, -1), K)

    topk_clses = (topk_ind / K).int()
    # print("topk_clses.shape", topk_clses.shape)  # 1×128

    topk_inds = _gather_feat(topk_inds.view(N, -1, 1), topk_ind).view(N, K)  # 1×128×1 -> 1×128?
    topk_ys = _gather_feat(topk_ys.view(N, -1, 1), topk_ind).view(N, K)
    topk_xs = _gather_feat(topk_xs.view(N, -1, 1), topk_ind).view(N, K)

    # 计算每个类别对应的topk索引
    cls_inds_masks = torch.full((num_classes, K), False, dtype=torch.bool).to(topk_inds.device)
    for cls_id in range(num_classes):
        inds_masks = topk_clses==cls_id
        # cls_topk_inds = topk_inds[inds_masks]
        cls_inds_masks[cls_id] = inds_masks

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs, cls_inds_masks


def mot_decode(heatmap,
               wh,
               reg=None,
               num_classes=2,
               cat_spec_wh=False,
               K=100):
    """
    :param heatmap:
    :param wh:
    :param reg:
    :param num_classes:
    :param cat_spec_wh:
    :param K:
    :return:
    """
    N, C, H, W = heatmap.size()  # N×C×H×W

    # heat = torch.sigmoid(heat)
    # perform nms(max pool) on heat-map
    heatmap = _max_pool(heatmap)  # 默认应用3×3max pooling操作, 检测目标数变为feature map的1/9

    # 根据heat-map取topK
    scores, inds, classes, ys, xs, cls_inds_masks = _topk(heatmap=heatmap, K=K, num_classes=num_classes)

    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(N, K, 2)
        xs = xs.view(N, K, 1) + reg[:, :, 0:1]
        ys = ys.view(N, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(N, K, 1) + 0.5
        ys = ys.view(N, K, 1) + 0.5

    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(N, K, C, 2)
        clses_ind = classes.view(N, K, 1, 1).expand(N, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(N, K, 2)
    else:
        wh = wh.view(N, K, 2)

    classes = classes.view(N, K, 1).float()  # 目标类别
    scores = scores.view(N, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] * 0.5,   # left    x1
                        ys - wh[..., 1:2] * 0.5,   # top     y1
                        xs + wh[..., 0:1] * 0.5,   # right   x2
                        ys + wh[..., 1:2] * 0.5],  # down    y2
                       dim=2)
    detections = torch.cat([bboxes, scores, classes], dim=2)

    return detections, inds, cls_inds_masks
