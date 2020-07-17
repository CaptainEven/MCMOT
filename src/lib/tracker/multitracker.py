from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from lib.models import *
from lib.models.decode import mot_decode
from lib.models.model import create_model, load_model
from lib.models.utils import _tranpose_and_gather_feat
from lib.tracker import matching
from lib.tracking_utils.kalman_filter import KalmanFilter
from lib.tracking_utils.log import logger
from lib.tracking_utils.utils import *
from lib.utils.post_process import ctdet_post_process
from .basetrack import BaseTrack, TrackState

# class name and class id mapping
cls2id = {
    'car': 0,
    'bicycle': 1,
    'person': 2,
    'cyclist': 3,
    'tricycle': 4
}

id2cls = {
    0: 'car',
    1: 'bicycle',
    2: 'person',
    3: 'cyclist',
    4: 'tricycle'
}


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buff_size=30):
        """
        :param tlwh:
        :param score:
        :param temp_feat:
        :param buff_size:
        """

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buff_size)  # 指定了限制长度
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * \
                               self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0

            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def reset_track_id(self):
        self.reset_track_count()

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_track.tlwh))

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean,
                                                               self.covariance,
                                                               self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()  # numpy中的.copy()是深拷贝
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


# rewrite a post processing(without using affine matrix)
def map2orig(dets, h_out, w_out, h_orig, w_orig, num_classes):
    """
    :param dets:
    :param h_out:
    :param w_out:
    :param h_orig:
    :param w_orig:
    :param num_classes:
    :return: dict of detections(key: cls_id)
    """

    def get_padding():
        """
        :return: pad_1, pad_2, pad_type('pad_x' or 'pad_y'), new_shape(w, h)
        """
        ratio_x = float(w_out) / w_orig
        ratio_y = float(h_out) / h_orig
        ratio = min(ratio_x, ratio_y)
        new_shape = (round(w_orig * ratio), round(h_orig * ratio))  # new_w, new_h

        pad_x = (w_out - new_shape[0]) * 0.5  # width padding
        pad_y = (h_out - new_shape[1]) * 0.5  # height padding
        top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
        left, right = round(pad_x - 0.1), round(pad_x + 0.1)
        if ratio == ratio_x:  # pad_y
            return top, bottom, 'pad_y', new_shape
        else:  # pad_x
            return left, right, 'pad_x', new_shape

    pad_1, pad_2, pad_type, new_shape = get_padding()

    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])  # default: 1×128×6
    dets = dets[0]  # 128×6

    dets_dict = {}

    if pad_type == 'pad_x':
        dets[:, 0] = (dets[:, 0] - pad_1) / new_shape[0] * w_orig  # x1
        dets[:, 2] = (dets[:, 2] - pad_1) / new_shape[0] * w_orig  # x2
        dets[:, 1] = dets[:, 1] / h_out * h_orig  # y1
        dets[:, 3] = dets[:, 3] / h_out * h_orig  # y2
    else:  # 'pad_y'
        dets[:, 0] = dets[:, 0] / w_out * w_orig  # x1
        dets[:, 2] = dets[:, 2] / w_out * w_orig  # x2
        dets[:, 1] = (dets[:, 1] - pad_1) / new_shape[1] * h_orig  # y1
        dets[:, 3] = (dets[:, 3] - pad_1) / new_shape[1] * h_orig  # y2

    classes = dets[:, -1]
    for cls_id in range(num_classes):
        inds = (classes == cls_id)
        dets_dict[cls_id] = dets[inds, :]

    return dets_dict


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        # if opt.gpus[0] >= 0:
        #     opt.device = torch.device('cuda')
        # else:
        #     opt.device = torch.device('cpu')

        # ----- init model
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)  # load specified checkpoint
        self.model = self.model.to(opt.device)
        self.model.eval()

        # ----- track_lets
        self.tracked_stracks_dict = defaultdict(list)  # value type: list[STrack]
        self.lost_stracks_dict = defaultdict(list)  # value type: list[STrack]
        self.removed_stracks_dict = defaultdict(list)  # value type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 10.0 * opt.track_buffer)  # int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = 128  # max objects per image
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        # 利用卡尔曼滤波过滤跟踪噪声
        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        """
        2D bbox检测结果后处理
        :param dets:
        :param meta:
        :return:
        """
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])  # default: 1×128×6

        # affine transform
        dets = ctdet_post_process(dets.copy(),
                                  [meta['c']], [meta['s']],
                                  meta['out_height'],
                                  meta['out_width'],
                                  self.opt.num_classes)

        # detection dict(cls_id as key)
        dets = dets[0]  # fetch the first image dets results(batch_size = 1 by default)

        return dets

    def merge_outputs(self, detections):
        """
        :param detections:
        :return:
        """
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate([detection[j] for detection in detections],
                                        axis=0).astype(np.float32)

        scores = np.hstack([results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]

        return results

    def update_detection(self, im_blob, img_0):
        """
        更新视频序列或图片序列的检测结果
        :rtype: dict
        :param im_blob:
        :param img_0:
        :return:
        """
        height, width = img_0.shape[0], img_0.shape[1]  # H, W of original input image
        net_height, net_width = im_blob.shape[2], im_blob.shape[3]  # H, W of net input

        c = np.array([width * 0.5, height * 0.5], dtype=np.float32)  # image center
        s = max(float(net_width) / float(net_height) * height, width) * 1.0

        h_out = net_height // self.opt.down_ratio
        w_out = net_width // self.opt.down_ratio

        # ----- get detections
        with torch.no_grad():
            dets_dict = defaultdict(list)

            # --- network output
            output = self.model.forward(im_blob)[-1]

            # --- detection outputs
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None

            # --- decode results of detection
            dets, inds, cls_inds_mask = mot_decode(heatmap=hm,
                                                   wh=wh,
                                                   reg=reg,
                                                   num_classes=self.opt.num_classes,
                                                   cat_spec_wh=self.opt.cat_spec_wh,
                                                   K=self.opt.K)

            # --- map to original image coordinate system
            # meta = {'c': c,
            #         's': s,
            #         'out_height': h_out,
            #         'out_width': w_out}
            # dets = self.post_process(dets, meta)  # using affine matrix
            dets = map2orig(dets, h_out, w_out, height, width, self.opt.num_classes)  # translate and scale
            # dets = self.merge_outputs([dets])

            # --- parse detections of each class
            for cls_id in range(self.opt.num_classes):  # cls_id start from index 0
                cls_dets = dets[cls_id]

                # filter out low conf score dets
                remain_inds = cls_dets[:, 4] > self.opt.conf_thres
                cls_dets = cls_dets[remain_inds]
                dets_dict[cls_id] = cls_dets

        return dets_dict

    # JDE跟踪器更新追踪状态
    def update_tracking(self, im_blob, img_0):
        """
        :param im_blob:
        :param img_0:
        :return:
        """
        # update frame id
        self.frame_id += 1

        # 记录跟踪结果
        # 记录跟踪结果: 默认只有一类, 修改为多类别, 用defaultdict(list)代替list
        # 以class id为key
        activated_starcks_dict = defaultdict(list)
        refind_stracks_dict = defaultdict(list)
        lost_stracks_dict = defaultdict(list)
        removed_stracks_dict = defaultdict(list)
        output_stracks_dict = defaultdict(list)

        height, width = img_0.shape[0], img_0.shape[1]  # H, W of original input image
        net_height, net_width = im_blob.shape[2], im_blob.shape[3]  # H, W of net input

        c = np.array([width * 0.5, height * 0.5], dtype=np.float32)
        s = max(float(net_width) / float(net_height) * height, width) * 1.0
        h_out = net_height // self.opt.down_ratio
        w_out = net_width // self.opt.down_ratio

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model.forward(im_blob)[-1]

            hm = output['hm'].sigmoid_()
            wh = output['wh']
            reg = output['reg'] if self.opt.reg_offset else None
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)  # L2 normalize

            #  检测和分类结果解析
            dets, inds, cls_inds_mask = mot_decode(heatmap=hm,
                                                   wh=wh,
                                                   reg=reg,
                                                   num_classes=self.opt.num_classes,
                                                   cat_spec_wh=self.opt.cat_spec_wh,
                                                   K=self.opt.K)

            # ----- get ReID feature vector by object class
            cls_id_feats = []  # topK feature vectors of each object class
            for cls_id in range(self.opt.num_classes):  # cls_id starts from 0
                # get inds of each object class
                cls_inds = inds[:, cls_inds_mask[cls_id]]

                # gather feats for each object class
                cls_id_feature = _tranpose_and_gather_feat(id_feature, cls_inds)  # inds: 1×128
                cls_id_feature = cls_id_feature.squeeze(0)  # n × FeatDim
                cls_id_feature = cls_id_feature.cpu().numpy()
                cls_id_feats.append(cls_id_feature)

        # 检测结果后处理
        # meta = {'c': c,
        #         's': s,
        #         'out_height': h_out,
        #         'out_width': w_out}
        # dets = self.post_process(dets, meta)  # using affine matrix
        # dets = self.merge_outputs([dets])

        dets = map2orig(dets, h_out, w_out, height, width, self.opt.num_classes)  # translate and scale

        # ----- 解析每个检测类别
        for cls_id in range(self.opt.num_classes):  # cls_id从0开始
            cls_dets = dets[cls_id]

            '''
            # 可视化中间的检测结果(每一类)
            for i in range(0, cls_dets.shape[0]):
                bbox = cls_dets[i][0:4]
                cv2.rectangle(img0,
                              (bbox[0], bbox[1]),  # left-top point
                              (bbox[2], bbox[3]),  # right-down point
                              [0, 255, 255],  # yellow
                              2)
                cv2.putText(img0,
                            id2cls[cls_id],
                            (bbox[0], bbox[1]),
                            cv2.FONT_HERSHEY_PLAIN,
                            1.3,
                            [0, 0, 255],  # red
                            2)
            cv2.imshow('{}'.format(id2cls[cls_id]), img0)
            cv2.waitKey(0)
            '''

            # 过滤掉score得分太低的dets
            remain_inds = cls_dets[:, 4] > self.opt.conf_thres
            cls_dets = cls_dets[remain_inds]
            cls_id_feature = cls_id_feats[cls_id][remain_inds]

            if len(cls_dets) > 0:
                '''Detections, tlbrs: top left bottom right score'''
                cls_detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], feat, buff_size=30)
                                  for (tlbrs, feat) in zip(cls_dets[:, :5], cls_id_feature)]
            else:
                cls_detections = []

            # reset the track ids for a different object class
            for track in cls_detections:
                track.reset_track_id()

            ''' Add newly detected tracklets to tracked_stracks'''
            unconfirmed_dict = defaultdict(list)
            tracked_stracks_dict = defaultdict(list)  # type: key(cls_id), value: list[STrack]
            for track in self.tracked_stracks_dict[cls_id]:
                if not track.is_activated:
                    unconfirmed_dict[cls_id].append(track)
                else:
                    tracked_stracks_dict[cls_id].append(track)

            ''' Step 2: First association, with embedding'''
            strack_pool_dict = defaultdict(list)
            strack_pool_dict[cls_id] = joint_stracks(tracked_stracks_dict[cls_id], self.lost_stracks_dict[cls_id])

            # Predict the current location with KF
            # for strack in strack_pool:
            STrack.multi_predict(strack_pool_dict[cls_id])
            dists = matching.embedding_distance(strack_pool_dict[cls_id], cls_detections)
            dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool_dict[cls_id], cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

            for i_tracked, i_det in matches:
                track = strack_pool_dict[cls_id][i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(cls_detections[i_det], self.frame_id)
                    activated_starcks_dict[cls_id].append(track)  # for multi-class
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks_dict[cls_id].append(track)

            ''' Step 3: Second association, with IOU'''
            cls_detections = [cls_detections[i] for i in u_detection]
            r_tracked_stracks = [strack_pool_dict[cls_id][i]
                                 for i in u_track if strack_pool_dict[cls_id][i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, cls_detections)
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

            for i_tracked, i_det in matches:
                track = r_tracked_stracks[i_tracked]
                det = cls_detections[i_det]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks_dict[cls_id].append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks_dict[cls_id].append(track)

            for it in u_track:
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks_dict[cls_id].append(track)

            '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
            cls_detections = [cls_detections[i] for i in u_detection]
            dists = matching.iou_distance(unconfirmed_dict[cls_id], cls_detections)
            matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
            for i_tracked, i_det in matches:
                unconfirmed_dict[cls_id][i_tracked].update(cls_detections[i_det], self.frame_id)
                activated_starcks_dict[cls_id].append(unconfirmed_dict[cls_id][i_tracked])
            for it in u_unconfirmed:
                track = unconfirmed_dict[cls_id][it]
                track.mark_removed()
                removed_stracks_dict[cls_id].append(track)

            """ Step 4: Init new stracks"""
            for i_new in u_detection:
                track = cls_detections[i_new]

                if track.score < self.det_thresh:
                    continue

                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks_dict[cls_id].append(track)

            """ Step 5: Update state"""
            for track in self.lost_stracks_dict[cls_id]:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks_dict[cls_id].append(track)

            # print('Ramained match {} s'.format(t4-t3))
            self.tracked_stracks_dict[cls_id] = [t for t in self.tracked_stracks_dict[cls_id] if
                                                 t.state == TrackState.Tracked]
            self.tracked_stracks_dict[cls_id] = joint_stracks(self.tracked_stracks_dict[cls_id],
                                                              activated_starcks_dict[cls_id])
            self.tracked_stracks_dict[cls_id] = joint_stracks(self.tracked_stracks_dict[cls_id],
                                                              refind_stracks_dict[cls_id])
            self.lost_stracks_dict[cls_id] = sub_stracks(self.lost_stracks_dict[cls_id],
                                                         self.tracked_stracks_dict[cls_id])
            self.lost_stracks_dict[cls_id].extend(lost_stracks_dict[cls_id])
            self.lost_stracks_dict[cls_id] = sub_stracks(self.lost_stracks_dict[cls_id],
                                                         self.removed_stracks_dict[cls_id])
            self.removed_stracks_dict[cls_id].extend(removed_stracks_dict[cls_id])
            self.tracked_stracks_dict[cls_id], self.lost_stracks_dict[cls_id] = remove_duplicate_stracks(
                self.tracked_stracks_dict[cls_id],
                self.lost_stracks_dict[cls_id])

            # get scores of lost tracks
            output_stracks_dict[cls_id] = [track for track in self.tracked_stracks_dict[cls_id] if track.is_activated]

            logger.debug('===========Frame {}=========='.format(self.frame_id))
            logger.debug('Activated: {}'.format(
                [track.track_id for track in activated_starcks_dict[cls_id]]))
            logger.debug('Refind: {}'.format(
                [track.track_id for track in refind_stracks_dict[cls_id]]))
            logger.debug('Lost: {}'.format(
                [track.track_id for track in lost_stracks_dict[cls_id]]))
            logger.debug('Removed: {}'.format(
                [track.track_id for track in removed_stracks_dict[cls_id]]))

        return output_stracks_dict


def joint_stracks(t_list_a, t_list_b):
    """
    join two track lists
    :param t_list_a:
    :param t_list_b:
    :return:
    """
    exists = {}
    res = []
    for t in t_list_a:
        exists[t.track_id] = 1
        res.append(t)
    for t in t_list_b:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()

    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)

    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]

    return resa, resb
