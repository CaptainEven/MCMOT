from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.decode import mot_decode
from lib.models.losses import FocalLoss
from lib.models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, \
    ArcMarginFc, CircleLoss, convert_label_to_similarity, McFocalLoss, GHMC
from lib.models.utils import _sigmoid, _tranpose_and_gather_feat
from lib.utils.post_process import ctdet_post_process

from .base_trainer import BaseTrainer


# 损失函数的定义
class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None  # L1 loss or smooth l1 loss
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg  # box size loss

        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID

        # 唯一包含可学习参数的层: 用于Re-ID的全连接层
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        # self.TriLoss = TripletLoss()

        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))  # 检测的损失缩放系数
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))  # track id分类的损失缩放系数

    def forward(self, outputs, batch):
        """
        :param outputs:
        :param batch:
        :return:
        """
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss = 0.0, 0.0, 0.0, 0.0  # 初始化4个loss为0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            # 计算heatmap loss
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                             batch['dense_wh'] * batch['dense_wh_mask']) /
                                mask_weight) / opt.num_stacks
                else:  # 计算box尺寸的L1/Smooth L1 loss
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:  # 计算box中心坐标偏移的L1 loss
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            # 检测目标id分类的交叉熵损失
            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()  # 只有有目标的像素才计算id loss
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]  # 有目标的track id
                id_output = self.classifier.forward(id_head).contiguous()  # 用于检测目标分类的最后一层是FC?
                id_loss += self.IDLoss(id_output, id_target)
                # id_loss += self.IDLoss(id_output, id_target) + self.TriLoss(id_head, id_target)

        # loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.id_weight * id_loss

        det_loss = opt.hm_weight * hm_loss \
                   + opt.wh_weight * wh_loss \
                   + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss \
               + torch.exp(-self.s_id) * id_loss \
               + (self.s_det + self.s_id)
        loss *= 0.5
        # print(loss, hm_loss, wh_loss, off_loss, id_loss)

        loss_stats = {'loss': loss,
                      'hm_loss': hm_loss,
                      'wh_loss': wh_loss,
                      'off_loss': off_loss,
                      'id_loss': id_loss}
        return loss, loss_stats


# 损失函数的定义
class McMotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(McMotLoss, self).__init__()

        self.opt = opt

        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None  # L1 loss or smooth l1 loss
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg  # box size loss

        # @even: Test additional loss functions for re-id
        # self.circle_loss = CircleLoss(m=0.25, gamma=80)
        # self.ghm_c = GHMC(bins=30)  # GHM_C loss for multi-class classification(For ReID)

        if opt.id_weight > 0:
            self.emb_dim = opt.reid_dim

            # @even: 用nID_dict取代nID, 用于MCMOT(multi-class multi-object tracking)训练
            self.nID_dict = opt.nID_dict

            # 包含可学习参数的层: 用于Re-ID的全连接层
            # @even: 为每个需要ReID的类别定义一个分类器
            self.classifiers = nn.ModuleDict()  # 使用ModuleList或ModuleDict才可以自动注册参数
            # self.focal_loss_dict = nn.ModuleDict()
            for cls_id, nID in self.nID_dict.items():
                # 选择一: 使用普通的全连接层
                self.classifiers[str(cls_id)] = nn.Linear(self.emb_dim, nID)  # FC layers

                # 选择二: 使用Arc margin全连接层
                # self.classifiers[str(cls_id)] = ArcMarginFc(self.emb_dim, nID, self.opt.device, 0.3)

                # 选择三: 使用Focal loss
                # self.focal_loss_dict[str(cls_id)] = McFocalLoss(nID, self.opt.device)

            # using CE loss to do ReID classification
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
            # self.TriLoss = TripletLoss()

            # @even: 为每个需要ReID的类别定义一个embedding scale
            self.emb_scale_dict = dict()
            for cls_id, nID in self.nID_dict.items():
                self.emb_scale_dict[cls_id] = math.sqrt(2) * math.log(nID - 1)

            # track reid分类的损失缩放系数
            self.s_id = nn.Parameter(-1.05 * torch.ones(1))  # -1.05

        # scale factor of detection loss
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))

    def forward(self, outputs, batch):
        """
        :param outputs:
        :param batch:
        :return:
        """
        opt = self.opt

        # 初始化4个loss为0
        hm_loss, wh_loss, off_loss, reid_loss = 0.0, 0.0, 0.0, 0.0
        for s in range(opt.num_stacks):
            # ----- Detection loss
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            # --- heat-map loss
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks

            # --- box width and height loss
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                             batch['dense_wh'] * batch['dense_wh_mask']) / mask_weight) \
                               / opt.num_stacks
                else:  # box width and height using L1/Smooth L1 loss
                    wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                             batch['ind'], batch['wh']) / opt.num_stacks

            # --- bbox center offset loss
            if opt.reg_offset and opt.off_weight > 0:  # offset using L1 loss
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            # ----- ReID loss: only process the class requiring ReID
            if opt.id_weight > 0:  # if ReID is needed
                cls_id_map = batch['cls_id_map']

                # 遍历每一个需要ReID的检测类别, 计算ReID损失
                for cls_id, id_num in self.nID_dict.items():
                    inds = torch.where(cls_id_map == cls_id)
                    if inds[0].shape[0] == 0:
                        # print('skip class id', cls_id)
                        continue

                    # --- 取cls_id对应索引处的特征向量
                    cls_id_head = output['id'][inds[0], :, inds[2], inds[3]]
                    cls_id_head = self.emb_scale_dict[cls_id] * F.normalize(cls_id_head)  # n × emb_dim

                    # --- 获取target类别
                    cls_id_target = batch['cls_tr_ids'][inds[0], cls_id, inds[2], inds[3]]

                    # ---分类结果
                    # normal FC layers
                    cls_id_pred = self.classifiers[str(cls_id)].forward(cls_id_head).contiguous()

                    # 使用Arc margin全连接层
                    # cls_id_pred = self.classifiers[str(cls_id)].forward(cls_id_head, cls_id_target).contiguous()

                    # --- 累加每一个检测类别的ReID loss
                    # 选择一: 使用交叉熵优化ReID
                    # print('\nNum objects:'); print(cls_id_target.nelement())
                    reid_loss += self.ce_loss(cls_id_pred, cls_id_target) / float(cls_id_target.nelement())

                    # 选择二: 使用Circle loss优化ReID
                    # reid_loss += self.circle_loss(*convert_label_to_similarity(cls_id_pred, cls_id_target))

                    # 选择三: 使用ce_loss + triplet loss优化ReID
                    # reid_loss += self.ce_loss(cls_id_pred, cls_id_target) + self.TriLoss(cls_id_head, cls_id_target)

                    # 选择三: Focal loss
                    # reid_loss += self.focal_loss_dict[str(cls_id)](cls_id_pred, cls_id_target)

                    # 选择四: 使用GHM loss
                    # target = torch.zeros_like(cls_id_pred)
                    # target.scatter_(1, cls_id_target.view(-1, 1).long(), 1)
                    # label_weight = torch.ones_like(cls_id_pred)
                    # reid_loss += self.ghm_c.forward(cls_id_pred, target, label_weight)

        # loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.id_weight * id_loss

        det_loss = opt.hm_weight * hm_loss \
                   + opt.wh_weight * wh_loss \
                   + opt.off_weight * off_loss

        if opt.id_weight > 0:
            loss = torch.exp(-self.s_det) * det_loss \
                   + torch.exp(-self.s_id) * reid_loss \
                   + (self.s_det + self.s_id)
        else:
            loss = torch.exp(-self.s_det) * det_loss \
                   + self.s_det

        loss *= 0.5
        # print(loss, hm_loss, wh_loss, off_loss, id_loss)

        if opt.id_weight > 0:
            loss_stats = {'loss': loss,
                          'hm_loss': hm_loss,
                          'wh_loss': wh_loss,
                          'off_loss': off_loss,
                          'id_loss': reid_loss}
        else:
            loss_stats = {'loss': loss,
                          'hm_loss': hm_loss,
                          'wh_loss': wh_loss,
                          'off_loss': off_loss}  # only exists det loss

        return loss, loss_stats


# 核心训练类
class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        if opt.id_weight > 0:  # tracking loss including det loss and re-id loss
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        else:  # only det loss
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']

        # loss = MotLoss(opt)
        loss = McMotLoss(opt)  # multi-class multi-object tracking loss

        return loss_states, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(heatmap=output['hm'],
                          wh=output['wh'],
                          reg=reg,
                          cat_spec_wh=self.opt.cat_spec_wh,
                          K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

        dets_out = ctdet_post_process(dets.copy(),
                                      batch['meta']['c'].cpu().numpy(),  # center
                                      batch['meta']['s'].cpu().numpy(),  # scale
                                      output['hm'].shape[2],  # height
                                      output['hm'].shape[3],  # width
                                      output['hm'].shape[1])  # num_classes

        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
