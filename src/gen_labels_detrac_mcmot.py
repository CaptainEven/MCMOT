import os
import shutil
import copy
import cv2
import xml.etree.ElementTree as ET
from collections import defaultdict
from tqdm import tqdm
from gen_mcmot_for_detect import target_types, classes, bbox_format


def preprocess(src_root, dst_root):
    """
    :param src_root:
    :param dst_root:
    :return:
    """
    if not os.path.isdir(src_root):
        print("[Err]: invalid source root")
        return

    if not os.path.isdir(dst_root):
        os.makedirs(dst_root)
        print("{} made".format(dst_root))

    # 创建用于训练MOT的目录结构
    dst_img_dir_train = dst_root + '/images/train'
    dst_img_dir_test = dst_root + '/images/test'
    dst_labels_with_ids = dst_root + '/labels_with_ids'
    if not os.path.isdir(dst_img_dir_train):
        os.makedirs(dst_img_dir_train)
    if not os.path.isdir(dst_img_dir_test):
        os.makedirs(dst_img_dir_test)
    if not os.path.isdir(dst_labels_with_ids):
        os.makedirs(dst_labels_with_ids)

    # 遍历src_root, 进一步完善训练目录并拷贝文件
    for x in os.listdir(src_root):
        x_path = src_root + '/' + x
        if os.path.isdir(x_path):
            for y in os.listdir(x_path):
                if y.endswith('.jpg'):
                    y_path = x_path + '/' + y
                    if os.path.isfile(y_path):
                        # 创建用于训练的图片目标目录
                        dst_img1_dir = dst_img_dir_train + '/' + x + '/img1'
                        if not os.path.isdir(dst_img1_dir):
                            os.makedirs(dst_img1_dir)

                        # 拷贝图片
                        shutil.copy(y_path, dst_img1_dir)
                        print('{} cp to {}'.format(y, dst_img1_dir))


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


def gen_labels(xml_root, img_root, label_root, viz_root=None):
    """
    解析xml(解析结果并可视化) + 生成labels
    :param xml_root:
    :param img_root:
    :param label_root:
    :param viz_root:
    :return:
    """
    if not (os.path.isdir(xml_root) and os.path.isdir(img_root)):
        print('[Err]: invalid dirs')
        return -1

    xml_f_paths = [xml_root + '/' + x for x in os.listdir(xml_root)]
    img_dirs = [img_root + '/' + x for x in os.listdir(img_root)]
    xml_f_paths.sort()  # 文件名自然排序
    img_dirs.sort()  # 目录名自然排序

    assert (len(xml_f_paths) == len(img_dirs))

    # 记录每一个序列开始的id, 初始化为0
    track_start_id = 0

    # 记录总的帧数
    frame_cnt = 0

    # 遍历每一个视频seq(每个seq对应一个xml文件和一个img_dir)
    for x, y in zip(xml_f_paths, img_dirs):
        if os.path.isfile(x) and os.path.isdir(y):
            if x.endswith('.xml'):  # 找到了xml原始标注文件
                sub_dir_name = os.path.split(y)[-1]
                if os.path.split(x)[-1][:-4] != sub_dir_name:
                    print('[Err]: xml file and dir not match')
                    continue

                # ----- 处理这个视频seq
                # 读取并解析xml
                tree = ET.parse(x)
                root = tree.getroot()
                seq_name = root.get('name')  # 视频序列名称(子目录名称)
                if seq_name != sub_dir_name:
                    print('[Warning]: xml file and dir not match')
                    continue
                print('Start processing seq {}...'.format(sub_dir_name))

                # 创建视频seq的训练标签子目录
                seq_label_root = label_root + '/' + seq_name + '/img1/'
                if not os.path.isdir(seq_label_root):
                    os.makedirs(seq_label_root)
                else:  # 如果已经存在, 则先清除原先的数据, 重新创建递归目录
                    shutil.rmtree(seq_label_root)
                    os.makedirs(seq_label_root)

                # 记录该视频seq的最大track_id
                seq_max_tar_id = 0

                # 创建seq_label_root
                seq_label_root = label_root + '/' + seq_name + '/img1'
                if not os.path.isdir(seq_label_root):
                    os.makedirs(seq_label_root)

                # 查找ignored_region(用于把原图对应矩形区域涂黑)
                ignor_region = root.find('ignored_region')

                # 记录该视频seq的ignored_region的所有box
                boxes = []
                for box_info in ignor_region.findall('box'):
                    box = [float(box_info.get('left')),
                           float(box_info.get('top')),
                           float(box_info.get('width')),
                           float(box_info.get('height'))]
                    # print('left {:.2f}, top {:.2f}, width {:.2f}, height {:.2f}'
                    #       .format(box[0], box[1], box[2], box[3]))
                    boxes.append(box)

                # 遍历每一帧
                for frame in root.findall('frame'):
                    # 更新帧数统计
                    frame_cnt += 1

                    target_list = frame.find('target_list')
                    targets = target_list.findall('target')
                    density = int(frame.get('density'))
                    if density != len(targets):  # 处理这一帧的每一个目标
                        print('[Err]: density not match @', frame)
                        return -1
                    # print('density {:d}'.format(density))

                    # ----- 处理当前帧
                    # 获取当前帧在这个视频seq中的frame id
                    f_id = int(frame.get('num'))

                    # ----- 读取视频seq的一帧, 将对应区域(ignore_region下box)涂黑
                    img_path = y + '/img1/img{:05d}.jpg'.format(f_id)
                    if not os.path.isfile(img_path):
                        print('[Err]: image file not exists!')
                        return -1
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # H×W×C
                    if img is None:  # channels: BGR
                        print('[Err]: read image failed!')
                        return -1

                    # 将图片涂黑后写入原训练目录路径
                    img = draw_ignore_regions(img, boxes)
                    cv2.imwrite(img_path, img)

                    # 如果可视化目录不为空, 进行可视化计算
                    if not (viz_root is None):
                        # 图片可视化目录和路径
                        viz_path = viz_root + '/' + seq_name + '_' + os.path.split(img_path)[-1]

                        # 深拷贝一份img数据作为可视化输出
                        img_viz = copy.deepcopy(img)

                    # 记录该帧的每一行label_str(对应一个检测or跟踪目标)
                    frame_label_strs = []

                    # 遍历这一帧中的每一个target(object)
                    for target in targets:
                        # 读取每一个目标, 追加的方式写入label_with_id
                        target_id = int(target.get('id'))

                        # 记录该视频seq最大的target id
                        if target_id > seq_max_tar_id:
                            seq_max_tar_id = target_id

                        # 记录该target(object)的track id(从1开始)
                        track_id = target_id + track_start_id

                        # 读取target对应的bbox
                        bbox_info = target.find('box')
                        bbox_left = float(bbox_info.get('left'))
                        bbox_top = float(bbox_info.get('top'))
                        bbox_width = float(bbox_info.get('width'))
                        bbox_height = float(bbox_info.get('height'))

                        # 读取target对应的属性(这里暂时只列出了感兴趣的属性)
                        attr_info = target.find('attribute')
                        vehicle_type = str(attr_info.get('vehicle_type'))
                        trunc_ratio = float(attr_info.get('truncation_ratio'))

                        # 在这里计算可视化的结果
                        if not (viz_root is None):  # 如果可视化目录不为空
                            # 为target绘制bbox
                            pt_1 = (int(bbox_left + 0.5), int(bbox_top + 0.5))
                            pt_2 = (int(bbox_left + bbox_width), int(bbox_top + bbox_height))
                            cv2.rectangle(img_viz,
                                          pt_1,
                                          pt_2,
                                          (0, 255, 0),
                                          2)
                            # 绘制属性文字
                            veh_type_str = 'Vehicle type: ' + vehicle_type
                            veh_type_str_size = cv2.getTextSize(veh_type_str,
                                                                cv2.FONT_HERSHEY_PLAIN,
                                                                1.3,
                                                                1)[0]
                            cv2.putText(img_viz,
                                        veh_type_str,
                                        (pt_1[0],
                                         pt_1[1] + veh_type_str_size[1] + 8),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.3,
                                        [225, 255, 255],
                                        1)
                            tr_id_str = 'Vehicle ID: ' + str(track_id)
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
                            trunc_str = 'Trunc ratio {:.2f}'.format(trunc_ratio)
                            trunc_str_size = cv2.getTextSize(trunc_str, cv2.FONT_HERSHEY_PLAIN, 1.3, 1)[0]
                            cv2.putText(img_viz,
                                        trunc_str,
                                        (pt_1[0],
                                         pt_1[1] + veh_type_str_size[1]
                                         + tr_id_str_size[1] + trunc_str_size[1] + 8),
                                        cv2.FONT_HERSHEY_PLAIN,
                                        1.3,
                                        [225, 255, 255],
                                        1)

                        # ----- 输出label
                        # 计算bbox中心点坐标
                        bbox_center_x = bbox_left + bbox_width * 0.5
                        bbox_center_y = bbox_top + bbox_height * 0.5

                        # 对bbox进行归一化([0.0, 1.0])
                        bbox_center_x /= img.shape[1]  # W
                        bbox_center_y /= img.shape[0]  # H
                        bbox_width /= img.shape[1]  # W
                        bbox_height /= img.shape[0]  # H

                        # 组织label的内容, TODO: 优化IO, 硬盘读写一次, 每帧label生成完成才输出
                        # class_id, track_id, bbox_center_x, box_center_y, bbox_width, bbox_height
                        label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            track_id,
                            bbox_center_x,  # center_x
                            bbox_center_y,  # center_y
                            bbox_width,  # bbox_w
                            bbox_height)  # bbox_h
                        frame_label_strs.append(label_str)

                        # # 输出label
                        # label_f_path = seq_label_root + '/img{:05d}.txt'.format(f_id)
                        # with open(label_f_path, 'a') as f:
                        #     f.write(label_str)

                    # 输出可视化结果
                    if not (viz_root is None):  # 如果可视化目录不为空
                        cv2.imwrite(viz_path, img_viz)

                    # ----- 这一帧的targets解析结束才输出一次
                    # 输出label
                    label_f_path = seq_label_root + '/img{:05d}.txt'.format(f_id)
                    with open(label_f_path, 'w') as f:
                        for label_str in frame_label_strs:
                            f.write(label_str)

                    # print('frame {} in seq {} processed done'.format(f_id, seq_name))

                # 处理完成该视频seq, 更新track_start_id
                print('Seq {} start track id: {:d}, has {:d} tracks'
                      .format(seq_name, track_start_id + 1, seq_max_tar_id))
                track_start_id += seq_max_tar_id
                print('Processing seq {} done.\n'.format(sub_dir_name))

    print('Total {:d} frames'.format(frame_cnt))


def add_new_train_data(part_train_f_path,
                       data_root,
                       dot_train_f_path,
                       dataset_prefix):
    """
    :param part_train_f_path:
    :param data_root:
    :param dot_train_f_path:
    :param dataset_prefix:
    :return:
    """
    if not os.path.isfile(part_train_f_path):
        print('[Err]: invalid src part dot train file.')
        return

    if not os.path.isfile(dot_train_f_path):
        print('[Err]: invalid final dot train file.')
        return

    if not os.path.isdir(data_root):
        print('[Err]: invalid data root.')
        return

    # 判断是否存在指定的目录结构(images目录和labels_with_ids目录)
    dst_img_root = data_root + '/images'
    dst_txt_root = data_root + '/labels_with_ids'
    if not (os.path.isdir(dst_img_root) and os.path.isdir(dst_txt_root)):
        print('[Err]: dst img root or txt root not exists!')
        return

    # 训练数据条数, 类别目标数, 计数
    item_cnt = 0
    class_cnt_dict = defaultdict(int)

    # 添加.train文件: 以追加的方式添加到已存在的.train文件
    train_f_h = open(dot_train_f_path, 'a', encoding='utf-8')

    with open(part_train_f_path, 'r', encoding='utf-8') as r_h:
        for line in r_h.readlines():

            src_img_path = line.strip()
            if not os.path.isfile(src_img_path):
                print('[Warning]: invalid image file path {}'.format(line))
                continue

            line = os.path.split(src_img_path)
            src_img_dir = line[0]
            img_name = line[-1]

            src_xml_path = src_img_dir.replace('JPEGImages', 'Annotations') + '/' + img_name.replace('.jpg', '.xml')
            if not os.path.isfile(src_xml_path):
                print('[Warning]: invalid xml file path {}'.format(src_xml_path))
                continue

            # 是否创建目标子目录
            src_sub_dir_name = src_img_dir.split('/')[-2]
            dst_img_sub_dir = dst_img_root + '/' + src_sub_dir_name
            dst_txt_sub_dir = dst_txt_root + '/' + src_sub_dir_name

            if not os.path.isdir(dst_img_sub_dir):
                os.makedirs(dst_img_sub_dir)
            if not os.path.isdir(dst_txt_sub_dir):
                os.makedirs(dst_txt_sub_dir)

            # 读取并解析xml
            tree = ET.parse(src_xml_path)
            root = tree.getroot()
            # print(root)

            mark_node = root.find('markNode')
            if mark_node is None:
                print('[Warning]: markNode not found.')
                continue

            label_obj_strs = []

            try:
                # 图片宽高
                w = int(root.find('width').text.strip())
                h = int(root.find('height').text.strip())
            except Exception as e:
                print('[Warning]: invalid (w, h)')
                print(e)
                continue

            # 更新item_cnt
            item_cnt += 1

            # 遍历该图片的每一个object
            for obj in mark_node.iter('object'):
                target_type = obj.find('targettype')
                cls_name = target_type.text
                if cls_name not in target_types:
                    print("=> " + cls_name + " is not in targetTypes list.")
                    continue

                # classes_c5(5类别的特殊处理)
                if cls_name == 'car_front' or cls_name == 'car_rear':
                    cls_name = 'car_fr'
                if cls_name == 'car':
                    car_type = obj.find('cartype').text
                    if car_type == 'motorcycle':
                        cls_name = 'bicycle'
                if cls_name == "motorcycle":
                    cls_name = "bicycle"
                if cls_name not in classes:
                    # print("=> " + cls_name + " is not in class list.")
                    continue
                if cls_name == 'non_interest_zone':
                    # print('Non interest zone.')
                    continue

                # 获取class_id
                cls_id = classes.index(cls_name)
                assert (0 <= cls_id < 5)

                # 更新class_cnt_dict
                class_cnt_dict[cls_name] += 1

                # 获取bounding box
                xml_box = obj.find('bndbox')
                box = (float(xml_box.find('xmin').text),
                       float(xml_box.find('xmax').text),
                       float(xml_box.find('ymin').text),
                       float(xml_box.find('ymax').text))

                # bounding box格式化: bbox([0.0, 1.0]): center_x, center_y, width, height
                bbox = bbox_format((w, h), box)
                if bbox is None:
                    print('[Warning]: bbox is err.')
                    continue

                # 生成检测对象的标签行: class_id, track_id, bbox_center_x, box_center_y, bbox_width, bbox_height
                obj_str = '{:d} 0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                    cls_id,  # class_id
                    bbox[0],  # center_x
                    bbox[1],  # center_y
                    bbox[2],  # bbox_w
                    bbox[3])  # bbox_h
                label_obj_strs.append(obj_str)

            # 拷贝图片到指定数据根目录下的子目录
            shutil.copy(src_img_path, dst_img_sub_dir)

            # 写入生成的txt格式的label到指定目标子目录
            txt_f_path = dst_txt_sub_dir + '/' + img_name.split('.')[0] + '.txt'
            with open(txt_f_path, 'w', encoding='utf-8') as w_h:
                for obj in label_obj_strs:
                    w_h.write(obj)
                print('{} written'.format(os.path.split(txt_f_path)[-1]))

            # 追加的方式写入.train文件
            train_str = str(dst_img_sub_dir + '/' + img_name).replace(dataset_prefix, '')
            train_f_h.write(train_str + '\n')

    # 释放final dot train file的句柄
    train_f_h.close()

    # 打印统计数据
    print('Total {:d} image files added to train dataset.'.format(item_cnt))
    for k, v in class_cnt_dict.items():
        print('Class {} contains {:d} items'.format(k, v))


def gen_dot_train_file(data_root, rel_path, out_root, f_name='detrac.train'):
    """
    生成.train文件
    :param data_root:
    :param rel_path:
    :param out_root:
    :param f_name:
    :return:
    """
    if not (os.path.isdir(data_root) and os.path.isdir(out_root)):
        print('[Err]: invalid root')
        return

    out_f_path = out_root + '/' + f_name
    cnt = 0
    with open(out_f_path, 'w') as f:
        root = data_root + rel_path
        seqs = [x for x in os.listdir(root)]
        # seqs.sort()
        seqs = sorted(seqs, key=lambda x: int(x.split('_')[-1]))
        for seq in tqdm(seqs):
            img_dir = root + '/' + seq  # + '/img1'
            img_list = [x for x in os.listdir(img_dir)]
            img_list.sort()
            for img in img_list:
                if img.endswith('.jpg'):
                    img_path = img_dir + '/' + img
                    if os.path.isfile(img_path):
                        item = img_path.replace(data_root + '/', '')
                        print(item)
                        f.write(item + '\n')
                        cnt += 1

    print('Total {:d} images for training'.format(cnt))


def find_file_with_suffix(root, suffix, f_list):
    """
    递归的方式查找特定后缀文件
    """
    for f in os.listdir(root):
        f_path = os.path.join(root, f)
        if os.path.isfile(f_path) and f.endswith(suffix):
            f_list.append(f_path)
        elif os.path.isdir(f_path):
            find_file_with_suffix(f_path, suffix, f_list)


def count_files(img_root, label_root):
    """
    统计总的图片个数和label txt文件个数
    :param img_root:
    :param label_root:
    :return:
    """
    img_file_list, label_f_list = [], []

    find_file_with_suffix(img_root, '.jpg', img_file_list)
    find_file_with_suffix(label_root, '.txt', label_f_list)

    print('Total {:d} image files'.format(len(img_file_list)))
    print('Total {:d} label(txt) files'.format(len(label_f_list)))


def clean_train_set(img_root, label_root):
    """
    清理图片个数与标签文件个数不匹配的问题
    :param img_root:
    :param label_root:
    :return:
    """
    if not (os.path.isdir(img_root) and os.path.isdir(label_root)):
        print('[Err]: incalid root!')
        return

    img_dirs = [img_root + '/' + x for x in os.listdir(img_root)]
    label_dirs = [label_root + '/' + x for x in os.listdir(label_root)]

    assert (len(img_dirs) == len(label_dirs))

    # 按视频seq名称排序
    img_dirs.sort()
    label_dirs.sort()

    for img_dir, label_dir in tqdm(zip(img_dirs, label_dirs)):
        # 一个couple一个couple的检查
        for img_name in os.listdir(img_dir + '/img1'):
            # print(img_name)
            txt_name = img_name.replace('.jpg', '.txt')
            txt_path = label_dir + '/img1/' + txt_name
            img_path = img_dir + '/img1/' + img_name
            if os.path.isfile(img_path) and os.path.isfile(txt_path):
                continue  # 两者同时存在, 无需处理
            elif os.path.isfile(img_path) and (not os.path.isfile(txt_path)):
                os.remove(img_path)
                print('{} removed.'.format(img_path))
            elif os.path.isfile(txt_path) and (not os.path.isfile(img_path)):
                os.remove(txt_path)
                print('{} removed.'.format(txt_path))


if __name__ == '__main__':
    # preprocess(src_root='/mnt/diskb/even/Insight-MVT_Annotation_Train',
    #            dst_root='/mnt/diskb/even/dataset/DETRAC')
    #
    # gen_labels(xml_root='/mnt/diskb/even/DETRAC-Train-Annotations-XML',
    #            img_root='/mnt/diskb/even/dataset/DETRAC/images/train',
    #            label_root='/mnt/diskb/even/dataset/DETRAC/labels_with_ids/train',
    #            viz_root='/mnt/diskb/even/viz_result')

    # gen_dot_train_file(data_root='/mnt/diskb/even/dataset',
    #                    rel_path='/MCMOT/images',
    #                    out_root='/mnt/diskb/even/MCMOT/src/data',
    #                    f_name='mcmot.train')

    add_new_train_data(part_train_f_path='/mnt/diskb/maqiao/multiClass/c5_hzpc_20200707/train.txt',
                       data_root='/mnt/diskb/even/dataset/MCMOT_DET',
                       dot_train_f_path='/mnt/diskb/even/MCMOT/src/data/mcmot_det.train',
                       dataset_prefix='/mnt/diskb/even/dataset/')

    # clean_train_set(img_root='/mnt/diskb/even/dataset/DETRAC/images/train',
    #                 label_root='/mnt/diskb/even/dataset/DETRAC/labels_with_ids/train')
    #
    # count_files(img_root='/mnt/diskb/even/dataset/DETRAC/images/train',
    #             label_root='/mnt/diskb/even/dataset/DETRAC/labels_with_ids')

    print('Done')
