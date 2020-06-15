from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds


def ctdet_post_process(dets, c, s, h, w, num_classes):
    """
    :param dets:
    :param c: center
    :param s: scale
    :param h: height
    :param w: width
    :param num_classes:
    :return:
    """
    # dets: batch x max_dets x dim
    # return 1-based class det dict

    ret = []
    for i in range(dets.shape[0]):  # 处理Batch中的每个数据
        top_preds = {}  # 字典(key: class_id(start from 1), value: obj_num×5)
        dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
        classes = dets[i, :, -1]

        for j in range(num_classes):
            inds = (classes == j)
            # top_preds[j + 1] = np.concatenate([
            #     dets[i, inds, :4].astype(np.float32),    # len(inds)×4
            #     dets[i, inds, 4:5].astype(np.float32)],  # len(inds)×1
            #                                   axis=1).tolist()  # 这里不输出类别(因为默认是一类)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),    # len(inds)×4
                dets[i, inds, 4:6].astype(np.float32)],  # len(inds)×1
                                              axis=1).tolist()  # 这里输出类别(因为有多类检测目标)
        ret.append(top_preds)

    return ret
