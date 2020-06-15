# encoding=utf-8
import xml.etree.ElementTree as ET
import pickle
import os
import re
import sys
import shutil
import math
from os import listdir, getcwd
from os.path import join
import cv2
import numpy as np

car = ["saloon_car", "suv", "van", "pickup"]
other = ["shop_truck", "unknown"]
bicycle = ["bicycle", "motorcycle"]

target_types = ["car",
               "car_front",
               "car_rear",
               "bicycle",
               "person",
               "cyclist",
               "tricycle",
               "motorcycle",
               "non_interest_zone",
               "non_interest_zones"]

classes_c9 = ["car",
              "truck",
              "waggon",
              "passenger_car",
              "other",
              "bicycle",
              "person",
              "cyclist",
              "tricycle",
              "non_interest_zone"]

classes_c6 = ['car',
              "bicycle",
              "person",
              "cyclist",
              "tricycle",
              "car_fr",
              "non_interest_zone",
              "non_interest_zones"]

classes_c5 = ['car',                 # 0
              "bicycle",             # 1
              "person",              # 2
              "cyclist",             # 3
              "tricycle",            # 4
              "non_interest_zone"]

# classes = classes_c6
classes = classes_c5  # 选择5类目标检测
class_num = len(classes) - 1  # 减1减的是non_interest_zone
car_fr = ["car_front", "car_rear"]

nCount = 0


def bbox_format(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_min = box[0]
    x_max = box[1]
    y_min = box[2]
    y_max = box[3]

    if x_min < 0:
        x_min = 0
    if x_max < 0 or x_min >= size[0]:
        return None
    if x_max >= size[0]:
        x_max = size[0] - 1
    if y_min < 0:
        y_min = 0
    if y_max < 0 or y_min >= size[1]:
        return None
    if y_max >= size[1]:
        y_max = size[1] - 1

    # bbox中心点坐标
    x = (x_min + x_max) / 2.0
    y = (y_min + y_max) / 2.0

    # bbox宽高
    w = abs(x_max - x_min)
    h = abs(y_max - y_min)

    # bbox中心点坐标和宽高归一化到[0.0, 1.0]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    if w == 0 or h == 0:
        return None

    return (x, y, w, h)


def convert_annotation(img_path, xml_path, label_path, file_name):
    in_file = open(xml_path + '/' + file_name + '.xml')
    out_file = open(label_path + '/' + file_name + '.txt', 'w')
    xml_info = in_file.read()

    if xml_info.find('dataroot') < 0:
        print("Can not find dataroot")
        out_file.close()
        in_file.close()
        return [], []

    # xml_info = xml_info.decode('GB2312').encode('utf-8')
    # xml_info = xml_info.replace('GB2312', 'utf-8')

    try:
        root = ET.fromstring(xml_info)
    except(Exception, e):
        print("Error: cannot parse file")
        # n = raw_input()
        out_file.close()
        in_file.close()
        return [], []

    boxes_non = []
    poly_non = []
    # Count = 0
    label_statis = [0 for i in range(class_num)]  # 每个类别目标的数量统计

    if root.find('markNode') != None:
        obj = root.find('markNode').find('object')
        if obj != None:
            w = int(root.find('width').text)
            h = int(root.find('height').text)
            print("w:%d,h%d" % (w, h))
            # print 'w=' + str(w) + ' h=' + str(h)

            for obj in root.iter('object'):
                target_type = obj.find('targettype')
                cls_name = target_type.text
                print(cls_name)
                if cls_name not in target_types:
                    print("********************************* " + cls_name +
                          " is not in targetTypes list *************************")
                    continue

                # # classes_c9
                # if cls == "car":
                #     cartype = obj.find('cartype').text
                #     # print(cartype)
                #     if cartype == 'motorcycle':
                #         cls = "bicycle"
                #     elif cartype == 'truck':
                #         cls = "truck"
                #     elif cartype == 'waggon':
                #         cls = 'waggon'
                #     elif cartype == 'passenger_car':
                #         cls = 'passenger_car'
                #     elif cartype == 'unkonwn' or cartype == "shop_truck":
                #         cls = "other"

                # classes_c5
                if cls_name == 'car_front' or cls_name == 'car_rear':
                    cls_name = 'car_fr'
                if cls_name == 'car':
                    car_type = obj.find('cartype').text
                    if car_type == 'motorcycle':
                        cls_name = 'bicycle'
                if cls_name == "motorcycle":
                    cls_name = "bicycle"
                if cls_name not in classes:
                    print("********************************* " + cls_name +
                          " is not in class list *************************")
                    continue

                cls_id = classes.index(cls_name)
                # print(cls_name, cls_id)
                cls_no = cls_id

                if cls_name == "non_interest_zones":  # 有个bug,non_interest_zones时为bndbox,胡老板已修复。
                    try:
                        xmlpoly = obj.find('polygonPoints').text
                        print('xml_poly:', xmlpoly)
                        poly_ = re.split('[,;]', xmlpoly)
                        poly_non.append(poly_)
                        continue
                    except:
                        continue

                # Count += 1
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text),
                     float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text),
                     float(xmlbox.find('ymax').text))  # 解析出来的bbox格式是left-up, right-down端点

                if cls_name == "non_interest_zone":
                    boxes_non.append(b)
                    continue

                # 统计每个类别目标的数量
                label_statis[cls_no] += 1

                # 转换bounding box格式: 归一化的值(center_x, center_y, w, h)
                bbox = bbox_format((w, h), b)

                if bbox is None:
                    print("++++++++++++++++++++++++++++++box is error++++++++++++++++++++")
                    # sleep(10)
                    continue

                # 写入label文件
                out_file.write(str(cls_no) + " " +
                               " ".join([str(a) for a in bbox]) + '\n')
                print(str(cls_no) + " " + " ".join([str(a) for a in bbox]))

    out_file.close()
    in_file.close()

    # if Count > 0:
    #     return 0
    # else:
    #     # if os.path.exists(labelpath+'/'+filename+'.txt'):
    #     #    os.remove(labelpath+'/'+filename+'.txt')
    #     return -1
    return poly_non, boxes_non, label_statis


if __name__ == "__main__":

    # rootdir = '/users/maqiao/mq/Data_checked/multiClass/multiClass0320'
    # root_path = "/users/maqiao/mq/Data_checked/multiClass/pucheng20191101"
    # rootdirs = [
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass0320',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass0507',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass0606',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass0704',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190808',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190814',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190822-1',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190822-3',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190823',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190826',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190827',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190827_1',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190830',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190830_1',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190830_2',
    # '/users/maqiao/mq/Data_checked/multiClass/multiClass190830_3'
    # "/users/maqiao/mq/Data_checked/multiClass/mark/houhaicui",
    # "/users/maqiao/mq/Data_checked/multiClass/mark/limingqing",
    # "/users/maqiao/mq/Data_checked/multiClass/mark/mayanzhuo",
    # "/users/maqiao/mq/Data_checked/multiClass/mark/quanqingfang",
    # "/users/maqiao/mq/Data_checked/multiClass/mark/shenjinyan",
    # "/users/maqiao/mq/Data_checked/multiClass/mark/wanglinan",
    # "/users/maqiao/mq/Data_checked/multiClass/mark/yangyanping",
    # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/houhaicui",
    # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/limingqing",
    # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/mayanzhuo",
    # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/quanqingfang",
    # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/shenjinyan",
    # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/wanglinan",
    # "/users/maqiao/mq/Data_checked/multiClass/duomubiao/yangyanping",
    # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20190912",
    # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20190920",
    # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20190925",
    # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20190930",
    # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20191011",
    # "/users/maqiao/mq/Data_checked/multiClass/tricycle_bigCar20191018",
    # "/users/maqiao/mq/Data_checked/multiClass/pucheng20191012",
    # "/users/maqiao/mq/Data_checked/multiClass/pucheng20191017",
    # "/users/maqiao/mq/Data_checked/multiClass/pucheng20191025",
    # "/users/maqiao/mq/Data_checked/multiClass/pucheng20191101"]
    # changsha_test_poly_nointer
    # /mnt/diskb/maqiao/multiClass/beijing20200110
    # /mnt/diskb/maqiao/multiClass/changsha20191224-2

    root_path = '/mnt/diskb/maqiao/multiClass/c5_puer_20200611'
    root_dirs = ["/mnt/diskb/maqiao/multiClass/c5_puer_20200611"]

    # root_path = '/users/duanyou/backup_c5/changsha_c5/test_new_chuiting'
    # rootdirs =  ["/users/duanyou/backup_c5/changsha_c5/test_new_chuiting"]

    # root_path = 'F:/mq1/test_data'
    # rootdirs  = [root_path+'/1']

    all_list_file = os.path.join(root_path, 'multiClass_train.txt')
    all_list = open(os.path.join(root_path, all_list_file), 'w')
    dir_num = len(root_dirs)
    for j, root_dir in enumerate(root_dirs):
        img_path = root_dir + '/' + "JPEGImages_ori"
        img_path_dst = root_dir + '/' + "JPEGImages"
        xml_path = root_dir + '/' + "Annotations"
        label_path = root_dir + '/' + "labels"

        if not os.path.exists(label_path):
            os.makedirs(label_path)
        if not os.path.exists(img_path_dst):
            os.makedirs(img_path_dst)

        list_file = open(root_dir + '/' + "train.txt", 'w')
        file_lists = os.listdir(img_path)
        file_num = len(file_lists)

        label_count = [0 for i in range(class_num)]
        for i, img_name in enumerate(file_lists):
            print("**************************************************************************************" +
                  str(i) + '/' + str(file_num) + '  ' + str(j) + '/' + str(dir_num))
            print(img_path + '/' + img_name)
            print(xml_path + '/' + img_name[:-4] + ".xml")

            if img_name.endswith('.jpg') and os.path.exists(xml_path + '/' + img_name[:-4] + ".xml"):
                if not os.path.exists(img_path):  # 没有对应的图片则跳过
                    continue

                poly_non, boxes_non, label_statistics = convert_annotation(img_path, xml_path, label_path, img_name[:-4])
                print('boxes_on:', boxes_non)

                if label_statistics == []:
                    continue

                label_count = [label_count[i] + label_statistics[i]
                               for i in range(class_num)]

                img_ori = img_path + '/' + img_name
                img = cv2.imread(img_ori)
                if img is None:
                    continue

                # 把不感兴趣区域替换成颜色随机的图像块
                is_data_ok = True
                if len(boxes_non) > 0:
                    for b in boxes_non:
                        x_min = int(min(b[0], b[1]))
                        x_max = int(max(b[0], b[1]))
                        y_min = int(min(b[2], b[3]))
                        y_max = int(max(b[2], b[3]))

                        if x_max > img.shape[1] or y_max > img.shape[0]:
                            is_data_ok = False
                            break

                        if x_min < 0:
                            x_min = 0
                        if y_min < 0:
                            y_min = 0
                        if x_max > img.shape[1] - 1:
                            x_max = img.shape[1] - 1
                        if y_max > img.shape[0] - 1:
                            y_max = img.shape[0] - 1

                        h = int(y_max - y_min)
                        w = int(x_max - x_min)

                        # 替换为马赛克
                        img[y_min:y_max, x_min:x_max, :] = np.random.randint(0, 255, (h, w, 3))

                # 把不感兴趣多边形区域替换成黑色
                if len(poly_non) > 0:
                    for poly in poly_non:
                        arr = []
                        i = 0

                        while i < len(poly) - 1:
                            arr.append([int(poly[i]), int(poly[i + 1])])
                            i = i + 2

                        arr = np.array(arr)
                        print('arr:', arr)
                        cv2.fillPoly(img, [arr], 0)

                if not is_data_ok:
                    continue

                img_dst = img_path_dst + '/' + img_name

                # 写入预处理后的图片
                print(img_dst)
                cv2.imwrite(img_dst, img)

                list_file.write(img_dst + '\n')
                all_list.write(img_dst + '\n')
            print("label_count ", label_count)

        list_file.close()
    all_list.close()
