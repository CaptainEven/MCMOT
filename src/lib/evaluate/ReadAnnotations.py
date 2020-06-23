# encoding=utf-8

import os
import xml.etree.ElementTree as ET


def Convert(size, box):
    """
    :param size:
    :param box:
    :return:
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = abs(box[1] - box[0])
    h = abs(box[3] - box[2])
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh

    return (x, y, w, h)


# 读取标注数据
def load_label(label_file, object_type):
    fl = open(label_file)
    cn = 0
    num = 0
    label_objs = []
    label_info = fl.read()
    if label_info.find('dataroot') < 0:
        print("Can not find dataroot")
        fl.close()
        return label_objs

    try:
        root = ET.fromstring(label_info)
    except(Exception, e):
        print("Error: cannot parse file")
        # n = raw_input()
        fl.close()
        return label_objs

    if root.find('markNode') != None:
        obj = root.find('markNode').find('object')
        if obj != None:
            w = int(root.find('width').text)
            h = int(root.find('height').text)
            # print("w:%d,h%d" % (w, h))
            for obj in root.iter('object'):
                target_type = obj.find('targettype').text
                car_type = obj.find('cartype').text
                if target_type == 'car_front' or target_type == 'car_rear' or target_type == 'car_fr':
                    target_type = 'fr'
                if target_type not in object_type and car_type not in object_type:
                    # print("********************************* "+str(targettype) + "is not in class list *************************")
                    continue

                # classes_c9
                # if targettype == "car":
                #     cartype = obj.find('cartype').text
                #     # print(cartype)
                #     if cartype == 'motorcycle':
                #         targettype = "bicycle"
                #     elif cartype == 'truck':
                #         targettype = "truck" 
                #     elif cartype == 'waggon':
                #         targettype = 'waggon'
                #     elif cartype == 'passenger_car':
                #         targettype = 'passenger_car'
                #     elif cartype == 'unkonwn' or cartype == "shop_truck":
                #         targettype = "other"

                # classes_c5
                if target_type == 'car':
                    car_type = obj.find('cartype').text
                    if car_type == 'motorcycle':
                        target_type = 'bicycle'
                if target_type == "motorcycle":
                    target_type = "bicycle"

                xml_box = obj.find('bndbox')
                b = (float(xml_box.find('xmin').text),
                     float(xml_box.find('xmax').text),
                     float(xml_box.find('ymin').text),
                     float(xml_box.find('ymax').text))
                bb = Convert((w, h), b)

                obj = [target_type, float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
                # print(obj)
                label_objs.append(obj)

    return label_objs


if __name__ == "__main__":
    label_file = '/mnt/diskb/maqiao/multiClass/test_c6/Annotations/1_5_1.xml'
    object_types = ['car', 'bicycle', 'person', 'cyclist', 'tricycle', 'fr', ]

    objs = load_label(label_file, object_types)
    print(objs)
