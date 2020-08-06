# encoding=utf-8

import os
import copy
import numpy as np
import cv2
# import shutil
from collections import defaultdict
from tqdm import tqdm

# ignored regions (0),

# ----------------------1~10类是我们需要检测和跟踪的目标
# pedestrian      (1),  --> 0
# people          (2),  --> 1
# bicycle         (3),  --> 2
# car             (4),  --> 3
# van             (5),  --> 4
# truck           (6),  --> 5
# tricycle        (7),  --> 6
# awning-tricycle (8),  --> 7
# bus             (9),  --> 8
# motor           (10), --> 9
# ----------------------

# others          (11)

# We need 10 classes to detect and tracking
cls2id = {
    'pedestrian': 0,
    'people': 1,
    'bicycle': 2,
    'car': 3,
    'van': 4,
    'truck': 5,
    'tricycle': 6,
    'awning-tricycle': 7,
    'bus': 8,
    'motor': 9
}

id2cls = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor'
}


def draw_ignore_regions(img, boxes):
    """
    输入图片ignore regions涂黑
    :param img: opencv(numpy array): H×W×C
    :param boxes: a list of boxes: left(box[0]), top(box[1]), width(box[2]), height(box[3])
    :return:
    """
    if img is None:
        print('[Err]: Input image is none!')
        return -1

    for box in boxes:
        box = list(map(lambda x: int(x + 0.5), box))  # 四舍五入
        img[box[1]: box[1] + box[3], box[0]: box[0] + box[2]] = [0, 0, 0]

    return img


def gen_track_dataset(src_root, dst_root, viz_root=None):
    """
    :param src_root:
    :param dst_root:
    :param viz_root:
    :return:
    """
    if not os.path.isdir(src_root):
        print('[Err]: invalid sr dir.')
        return

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)

    dst_img_root = dst_root + '/images'
    dst_txt_root = dst_root + '/labels_with_ids'
    if not os.path.isdir(dst_img_root):
        os.makedirs(dst_img_root)
    if not os.path.isdir(dst_txt_root):
        os.makedirs(dst_txt_root)

    # 记录每一个序列开始的id, 初始化为0
    # track_start_id = 0
    track_start_id_dict = defaultdict(int)  # 所有类别start id都从0开始
    for cls_id in id2cls.keys():
        track_start_id_dict[cls_id] = 0

    # 记录总的帧数
    frame_cnt = 0

    seq_names = [x for x in os.listdir(src_root + '/sequences')]
    seq_names.sort()

    # 遍历每一个视频序列
    for seq in tqdm(seq_names):
        print('Processing {}:'.format(seq))

        seq_img_dir = src_root + '/sequences/' + seq
        seq_txt_f_path = src_root + '/annotations/' + seq + '.txt'

        if not (os.path.isdir(seq_img_dir) and os.path.isfile(seq_txt_f_path)):
            print('[Warning]: invalid src img dir or invalid annotations file(txt).')
            continue

        # 创建目标子目录(图片目录和标签目录)
        dst_seq_img_dir = dst_img_root + '/' + seq
        if not os.path.isdir(dst_seq_img_dir):
            os.makedirs(dst_seq_img_dir)
        dst_seq_txt_dir = dst_txt_root + '/' + seq
        if not os.path.isdir(dst_seq_txt_dir):
            os.makedirs(dst_seq_txt_dir)

        # 记录该视频seq的最大track_id
        # seq_max_tar_id = 0
        seq_max_tra_id_dict = defaultdict(int)
        for k in id2cls.keys():
            seq_max_tra_id_dict[k] = 0

        # 视频序列
        seq_frame_names = os.listdir(seq_img_dir)
        seq_frame_names.sort()

        # 将该序列的标签文件读入二维数组
        with open(seq_txt_f_path, 'r', encoding='utf-8') as f_r:
            label_lines = f_r.readlines()
            label_n_lines = len(label_lines)
            seq_label_array = np.zeros((label_n_lines, 10), np.int32)

            # 解析该视频序列的每一帧
            for line_i, line in enumerate(label_lines):
                line = [int(x) for x in line.strip().split(',')]
                seq_label_array[line_i] = line

        # 记录该视频序列每一帧的ignore_regions和检测/跟踪目标
        seq_ignore_box_label = seq_label_array[seq_label_array[:, 7] == 0]
        seq_obj_boxes = seq_label_array[(seq_label_array[:, 7] > 0) & (seq_label_array[:, 7] < 11)]  # np条件索引
        seq_ignore_box_dict = defaultdict(list)
        seq_objs_label_dict = defaultdict(list)
        for label in seq_ignore_box_label:  # key: frame_id(start from 1)
            seq_ignore_box_dict[label[0]].append(label[2:6])
        for label in seq_obj_boxes:  # key: frame_id(start from 1)
            seq_objs_label_dict[label[0]].append(label)

        # 为此seq维护一个dict记录每个class对对应的target id
        seq_cls_target_ids_dict = defaultdict(list)
        tmp_ids_dict = defaultdict(set)
        for fr_id in seq_objs_label_dict.keys():  # 处理每一帧
            fr_labels = seq_objs_label_dict[fr_id]

            for label in fr_labels:
                cls_id = label[7] - 1
                target_id = label[1]
                # seq_cls_target_ids_dict[cls_id].append(target_id)  # key: cls_id
                tmp_ids_dict[cls_id].add(target_id)

        for cls_id in tmp_ids_dict.keys():
            track_ids = tmp_ids_dict[cls_id]
            # track_ids = set(track_ids)
            track_ids = list(track_ids)
            track_ids.sort()
            seq_cls_target_ids_dict[cls_id] = track_ids

        # 更新max_track_id
        for k, v in seq_cls_target_ids_dict.items():
            seq_max_tra_id_dict[k] = len(v)
        # print('Seq {}:'.format(seq))
        for k in id2cls.keys():
            print("{} max track id: {:d}, start id: {:d}"
                  .format(id2cls[k], seq_max_tra_id_dict[k], track_start_id_dict[k]))

        # 读取每一帧
        for fr_id in seq_objs_label_dict.keys():
            # -----
            fr_labels = seq_objs_label_dict[fr_id]

            # ----- 读取图片宽高
            fr_name = '{:07d}.jpg'.format(fr_id)
            fr_path = seq_img_dir + '/' + fr_name
            if not os.path.isfile(fr_path):
                print('[Err]: invalid image file {}.'.format(fr_path))
                continue

            # H×W×C: BGR
            img = cv2.imread(fr_path, cv2.IMREAD_COLOR)
            if img is None:
                print('[Err]: empty image.')
                continue

            H, W, C = img.shape

            # ----- 绘制ignore regions
            draw_ignore_regions(img, seq_ignore_box_dict[fr_id])

            # ----- 拷贝image到目标目录
            dst_img_path = dst_seq_img_dir + '/' + fr_name
            if not os.path.isfile(dst_img_path):
                cv2.imwrite(dst_img_path, img)  # 将绘制过ignore region的图片存入目标子目录
                # print('{} saved to {}'.format(fr_path, dst_seq_img_dir))

            # ----- 如果可视化目录不为空, 进行可视化计算
            if not (viz_root is None):
                # 图片可视化目录和路径
                viz_dir = viz_root + '/' + seq
                if not os.path.isdir(viz_dir):
                    os.makedirs(viz_dir)
                viz_path = viz_dir + '/' + fr_name

                # 深拷贝一份img数据作为可视化输出
                img_viz = copy.deepcopy(img)

            # ----- 生成label文件(txt)
            # 记录该帧的每一行label_str(对应一个检测or跟踪目标)
            fr_label_strs = []
            for label in fr_labels:
                # cls_id and cls_name
                obj_type = label[7]
                assert 0 < obj_type < 11
                cls_id = obj_type - 1  # 从0开始
                # cls_name = id2cls[cls_id]

                target_id = label[1]

                # 记录该target(object)的track id(从1开始: 标签中从0开始)
                track_id = seq_cls_target_ids_dict[cls_id].index(target_id) + 1 + track_start_id_dict[cls_id]
                # track_id = target_id

                bbox_left = label[2]
                bbox_top = label[3]
                bbox_width = label[4]
                bbox_height = label[5]

                score = label[6]
                truncation = label[8]  # no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% °´ 50%))
                occlusion = label[9]
                if occlusion > 1:  # heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).
                    # print('[Warning]: skip the bbox because of heavy occlusion')
                    continue

                # ----- 绘制该label(一个label是一张图的一个检测/跟踪目标): 在归一化之前
                if not (viz_root is None):  # 如果可视化目录不为空
                    # 为target绘制bbox
                    pt_1 = (int(bbox_left + 0.5), int(bbox_top + 0.5))
                    pt_2 = (int(bbox_left + bbox_width), int(bbox_top + bbox_height))
                    cv2.rectangle(img_viz,
                                  pt_1,
                                  pt_2,
                                  (0, 255, 0),
                                  2)

                    # 绘制类别文字
                    cls_str = id2cls[cls_id]
                    veh_type_str_size = cv2.getTextSize(cls_str,
                                                        cv2.FONT_HERSHEY_PLAIN,
                                                        1.3,
                                                        1)[0]
                    cv2.putText(img_viz,
                                cls_str,
                                (pt_1[0],
                                 pt_1[1] + veh_type_str_size[1] + 8),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.3,
                                [225, 255, 255],
                                1)

                    # 绘制track id
                    tr_id_str = str(track_id)
                    tr_id_str_size = cv2.getTextSize(tr_id_str,
                                                     cv2.FONT_HERSHEY_PLAIN,
                                                     1.3,
                                                     1)[0]
                    cv2.putText(img_viz,
                                tr_id_str,
                                (pt_1[0],
                                 pt_1[1] + veh_type_str_size[1] + tr_id_str_size[1] + 8),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.3,
                                [225, 255, 255],
                                1)

                # 计算bbox中心点坐标
                bbox_center_x = bbox_left + bbox_width * 0.5
                bbox_center_y = bbox_top + bbox_height * 0.5

                # 对bbox进行归一化([0.0, 1.0])
                bbox_center_x /= W
                bbox_center_y /= H
                bbox_width /= W
                bbox_height /= H

                # 组织label的内容, 每帧label生成完成才输出
                # class_id, track_id, bbox_center_x, box_center_y, bbox_width, bbox_height
                label_str = '{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    cls_id,
                    track_id,
                    bbox_center_x,
                    bbox_center_y,
                    bbox_width,
                    bbox_height)
                fr_label_strs.append(label_str)

            # ----- 输出可视化结果
            if not (viz_root is None):  # 如果可视化目录不为空
                cv2.imwrite(viz_path, img_viz)

            # ----- 这一帧的targets解析结束才输出一次
            # 输出label
            label_f_path = dst_seq_txt_dir + '/' + fr_name.replace('.jpg', '.txt')
            with open(label_f_path, 'w', encoding='utf-8') as f:
                for label_str in fr_label_strs:
                    f.write(label_str)
            # print('{} written.'.format(label_f_path))

            frame_cnt += 1

        # # 处理完成该视频seq, 更新track_start_id
        for cls_id in id2cls.keys():
            track_start_id_dict[cls_id] += seq_max_tra_id_dict[cls_id]
        print('Processing seq {} done.\n'.format(seq))

    print('Total {:d} frames'.format(frame_cnt))


if __name__ == '__main__':
    gen_track_dataset(src_root='/mnt/diskb/even/VisDrone2019-MOT-train',
                      dst_root='/mnt/diskb/even/dataset/VisDrone2019',
                      viz_root='/mnt/diskb/even/viz_result')
