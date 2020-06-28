# encoding=utf-8

import os
import cv2
import time
import numpy
import copy

import lib.evaluate.darknet as dn
import lib.evaluate.cmp_det_label_sf as cdl

from lib.evaluate.ReadAndSaveDarknetDetRes import read_det_res, save_det_res
from lib.evaluate.ReadAnnotations import load_label
from lib.evaluate.voc_eval import voc_eval


# 读取文件列表
def Load_file_list(files):
    fl = open(files, "r")
    file_lists = []
    while True:
        lines = fl.readlines()
        if len(lines) == 0:
            break
        # print(path_list)

        for line in lines:
            line = line.strip('\n')
            # ph = line.split("/")
            # file_name = ph[-1]
            # file_name = os.path.basename(line)
            # file_name = file_name.replace(".jpg", "")
            file_lists.append(line)
            # print(file_name)
        # print(path_lists)
    fl.close()
    return file_lists


def listdir(path):
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            continue
            # listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name


def img_path2label_path(img_path):
    """
    :param img_path:
    :return:
    """
    image_dir = os.path.dirname(img_path)
    p = image_dir.split('/')
    root_dir = "/".join(p[:-1])
    label_dir = os.path.join(root_dir, 'Annotations')
    image_name = os.path.basename(img_path)
    image_name = image_name.replace(".jpg", "")
    label_path = os.path.join(label_dir, image_name + '.xml')

    return label_path


def get_file_name(file_path):
    file_name = os.path.basename(file_path)
    p = file_name.split('.')
    name = ''
    for i in range(len(p) - 1):
        name += p[i]
    # file_name = p[]

    return name


def getMetaCfgName(file_path):
    # 寻找file_path的同文件夹里的.data文件
    p = os.path.dirname(file_path)
    for file in os.listdir(p):
        if '.data' in file:
            data_path = file
            data_path = p + '/' + data_path
        if 'test.cfg' in file:
            cfg_path = file
            cfg_path = p + '/' + cfg_path

    return data_path.encode('utf-8'), cfg_path.encode('utf-8')


def batch_detection():
    pass


def batch_analysis(weights_list_file,
                   img_list_file,
                   thresh,
                   iou_thresh,
                   result_dir):
    """
    :param weights_list_file:
    :param img_list_file:
    :param thresh:
    :param iou_thresh:
    :param result_dir:
    :return:
    """
    image_list = Load_file_list(img_list_file)
    image_num = len(image_list)
    weights_list = Load_file_list(weights_list_file)
    result = []
    for weights in weights_list:
        weights_name = get_file_name(weights)

        # print('weights_name: ',weights)

        meta_file, cfg_file = getMetaCfgName(weights)
        # meta = dn.load_meta(meta_file)
        # net = dn.load_net(cfg_file,bytes(weights,'utf-8'),0)

        # 选择对应的dn
        meta = dn.load_meta(meta_file)
        net = dn.load_net(cfg_file, bytes(weights, 'utf-8'), 0)

        object_type = [meta.names[i].decode('utf-8').strip() for i in range(meta.classes)]

        # @even: tmp modification
        weights_name = 'mcmot'

        result_path = os.path.join(result_dir, weights_name)
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        # @even: comment detections now
        # # detect result and save to text
        # time_all = 0
        # for j, img_path in enumerate(image_list):
        #     print('detect: ' + str(j + 1) + '/' + str(len(image_list)))
        #     label_path = img_path2label_path(img_path)
        #     image_name = get_file_name(img_path)
        #     det_save_path = os.path.join(result_path, image_name + '.txt')
        #     # det = dn.detect_ext(net, meta, bytes(image_path,'utf-8'),thresh)
        #
        #     # 选择对应的dn
        #     det, time_1 = dn.detect_ext(net, meta, bytes(img_path, 'utf-8'), thresh)
        #     time_all = time_all + time_1
        #
        #     # save detection result to text
        #     save_det_res(det, det_save_path, object_type)
        #     time.sleep(0.001)
        # print('xxxxxxxxxxx', 'FPS, ', len(image_list) / time_all)
        # # dn.free_net(net)

        # compare label and detection result
        for i, obj_type in enumerate(object_type):

            # if obj_type != 'fr':
            #     continue

            total_label = 0
            total_detect = 0
            total_corr = 0
            total_iou = 0
            cmp_result = []
            det_ = []
            anno_path = []

            det_all = [['name', 'obj_type', 'score', 0, 0, 0, 0]]  # 此处为xywh(中心), 应该变为xmin, ymin, xmax, ymax

            img_set_file = []
            for j, img_path in enumerate(image_list):
                label_path = img_path2label_path(img_path)
                image_name = get_file_name(img_path)
                img_set_file.append(image_name)
                img_save_path = os.path.join(result_path, image_name + '.jpg')
                det_save_path = os.path.join(result_path, image_name + '.txt')

                # detpath.append(det_save_path)
                anno_path.append(label_path)
                # print(img_save_path)
                label = []
                if os.path.exists(label_path):
                    label = load_label(label_path, object_type)

                # save detection result to text
                det = read_det_res(det_save_path)
                for d in det:
                    if d[0] > len(object_type) - 1:
                        d[0] = ' '
                        continue

                    d[0] = object_type[d[0]]  # 类别编号 -> 类别名称

                for d in det:
                    x_min = float(copy.deepcopy(d[2])) - float(copy.deepcopy(d[4])) * 0.5
                    y_min = float(copy.deepcopy(d[3])) - float(copy.deepcopy(d[5])) * 0.5
                    x_max = float(copy.deepcopy(d[2])) + float(copy.deepcopy(d[4])) * 0.5
                    y_max = float(copy.deepcopy(d[3])) + float(copy.deepcopy(d[5])) * 0.5

                    # ----- img_name  type  conf  x_min  y_min  x_max  y_max
                    d_ = [image_name, d[0], d[1], x_min, y_min, x_max, y_max]
                    det_.append(d_)

                if len(det_) != 0:
                    det_all = numpy.vstack((det_all, det_))
                det_ = []

                if i > 0:
                    img_path = img_save_path

                # print(j, image_path)
                img = cv2.imread(img_path)
                if img is None:
                    print("load image error&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    continue

                cmp_res = cdl.cmp_data(obj_type, det, label, thresh, iou_thresh, img)

                cmp_res.update({'image_name': image_name})
                total_corr += cmp_res['correct']
                total_iou += cmp_res['avg_iou'] * cmp_res['label_num']

                cmp_result.append(cmp_res)
                print(
                    "%s: %d/%d  label: %d   detect: %d   correct: %d   recall: %f   avg_iou: %f   accuracy: %f   precision: %f\n" %
                    (str(obj_type), j + 1, image_num, cmp_res['label_num'], cmp_res['detect_num'],
                     cmp_res['correct'], cmp_res['recall'], cmp_res['avg_iou'],
                     cmp_res['accuracy'], cmp_res['precision']))
                total_label += cmp_res['label_num']
                total_detect += cmp_res['detect_num']
                cv2.imwrite(img_save_path, img)
                img = []
                time.sleep(0.001)

            # 求出AP值
            # ap=0
            det_all = numpy.delete(det_all, 0, axis=0)
            det_obj_type = [obj for obj in det_all if obj[1] == obj_type]
            if len(det_obj_type) == 0:
                ap = 0
            else:
                ap = voc_eval(det_obj_type, anno_path, img_set_file, obj_type, iou_thresh)
            det_all = []

            # 数据集分析结果
            avg_recall = 0
            if total_label > 0:
                avg_recall = total_corr / float(total_label)
            avg_iou = 0
            if total_iou > 0:
                avg_iou = total_iou / total_label
            avg_acc = 0
            if total_label + total_detect - total_corr > 0:
                avg_acc = float(total_corr) / (total_label + total_detect - total_corr)
            avg_precision = 0
            if total_detect > 0:
                avg_precision = float(total_corr) / total_detect
            total_result = [total_label, total_detect, total_corr, avg_recall, avg_iou, avg_acc, avg_precision]
            cdl.ExportAnaRes(obj_type, cmp_result, total_result, img_path, result_path)
            print(
                "total_label: %d   total_detect: %d   total_corr: %d   recall: %f   average iou: %f   accuracy: %f   precision: %f ap: %f\n" % \
                (total_result[0], total_result[1], total_result[2], total_result[3], total_result[4], total_result[5],
                 total_result[6], ap))

            result.append([weights_name] + [obj_type] + total_result + [float(ap)])

        # 输出所有类别总的结果
        cdl.ExportAnaResAll(result, result_dir)
        time.sleep(0.001)


if __name__ == "__main__":

    dn.set_gpu(0)
    # weights_list_file = "/users/duanyou/c5/puer/weights.txt"
    # weights_list_file = "/users/duanyou/c5/v4_all_train/weights.txt"
    weights_list_file = "/users/duanyou/c5/v4_half_train/weights.txt"

    # yancheng_test
    # data_path = "/users/duanyou/c5/yancheng"
    # image_list_file = os.path.join(data_path,"test.txt")
    # result_dir = os.path.join("/users/duanyou/c5/results_new/results_yancheng/")
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    # batch_analysis(weights_list_file,image_list_file,0.20,0.45,result_dir)

    # # all_test
    # data_path = "/users/duanyou/c5/all_pretrain"
    # image_list_file = os.path.join(data_path,"test.txt")
    # result_dir = os.path.join("/users/duanyou/c5/results_new/results_all/")
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    # batch_analysis(weights_list_file,image_list_file,0.20,0.45,result_dir)

    # # changsha_test
    # data_path = "/users/duanyou/c5/changsha"
    # image_list_file = os.path.join(data_path,"test.txt")
    # result_dir = os.path.join("/users/duanyou/c5/results_new/results_changsha/")
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    # batch_analysis(weights_list_file,image_list_file,0.20,0.45,result_dir)

    # hezhoupucheng_test
    # data_path = "/users/duanyou/c5/hezhoupucheng"
    # image_list_file = os.path.join(data_path,"test.txt")
    # result_dir = os.path.join("/users/duanyou/c5/results_new/results_hezhoupucheng/")
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    # batch_analysis(weights_list_file,image_list_file,0.20,0.45,result_dir)

    # ----- puer_test
    data_path = "/users/duanyou/c5/puer"
    image_list_file = os.path.join(data_path, "test.txt")
    result_dir = os.path.join("/users/duanyou/c5/results_new/results_puer/")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    batch_analysis(weights_list_file, image_list_file, 0.20, 0.45, result_dir)

    # # some_img_test
    # data_path = "/users/duanyou/backup_c5/test_2"
    # image_list_file = os.path.join(data_path, "test.txt")
    # result_dir = os.path.join("/users/duanyou/backup_c5/test_2/result/")
    #
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    #
    # batch_analysis(weights_list_file, image_list_file, 0.20, 0.45, result_dir)
