import os
import xml.etree.ElementTree as ET

def Convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = abs(box[1] - box[0])
    h = abs(box[3] - box[2])
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

#读取标注数据
def LoadLabel(label_file, object_type):
    fl = open(label_file)
    cn = 0
    num = 0
    label_objs = []
    label_info = fl.read()
    if label_info.find('dataroot') < 0:
        print ("Can not find dataroot")
        fl.close()
        return label_objs

    try:
        root = ET.fromstring(label_info)
    except(Exception,e):
        print ("Error: cannot parse file")
        #n = raw_input()
        fl.close()
        return label_objs

    if root.find('markNode') != None:
        obj = root.find('markNode').find('object')
        if obj != None:
            w = int(root.find('width').text)
            h = int(root.find('height').text)
            #print("w:%d,h%d" % (w, h))
            for obj in root.iter('object'):
                targettype = obj.find('targettype').text
                cartype = obj.find('cartype').text
                if targettype == 'car_front' or targettype == 'car_rear' or targettype == 'car_fr':
                    targettype = 'fr'
                if targettype not in object_type and cartype not in object_type:
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
                if targettype == 'car':
                    cartype = obj.find('cartype').text
                    if cartype == 'motorcycle':
                        targettype = 'bicycle'
                if targettype == "motorcycle":
                    targettype = "bicycle"

                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),float(xmlbox.find('ymax').text))
                bb = Convert((w, h), b)
                obj = [targettype,float(bb[0]),float(bb[1]),float(bb[2]),float(bb[3])]
                #print(obj)
                label_objs.append(obj)
    return label_objs

if __name__ == "__main__":
    label_file = '/mnt/diskb/maqiao/multiClass/test_c6/Annotations/1_5_1.xml'
    object_types = ['car','bicycle','person','cyclist','tricycle','fr',]

    objs = LoadLabel(label_file,object_types)
    print(objs)