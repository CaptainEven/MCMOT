import os
import darknet as dn

import cv2
import time
import numpy
import copy

import cmp_det_label as cdl
from readAndSaveDarknetDetRes import readDetRes,saveDetRes
from readAnnotations import LoadLabel
from voc_eval import voc_eval

#读取文件列表
def LoadFileList(files):
    fl = open(files,"r")
    file_lists = []
    while True:
        lines = fl.readlines()
        if len(lines) == 0:
            break
        #print(path_list)

        for line in lines:
            line = line.strip('\n')
            # ph = line.split("/")
            # file_name = ph[-1]
            # file_name = os.path.basename(line)
            # file_name = file_name.replace(".jpg", "")
            file_lists.append(line)
            #print(file_name)
        #print(path_lists)
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

def imagePath2labelPath(image_path):
    image_dir = os.path.dirname(image_path)
    p = image_dir.split('/')
    root_dir = "/".join(p[:-1])
    label_dir = os.path.join(root_dir,'Annotations')
    image_name = os.path.basename(image_path)
    image_name = image_name.replace(".jpg", "")
    label_path = os.path.join(label_dir, image_name+'.xml')
    return label_path

def getFileName(file_path):
    file_name = os.path.basename(file_path)
    p = file_name.split('.')
    name = ''
    for i in range(len(p)-1):
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

def batch_analysis(weights_list_file, image_list_file, thresh, iou_thresh,result_dir):
    image_list = LoadFileList(image_list_file)
    image_num = len(image_list)
    weights_list = LoadFileList(weights_list_file)
    result = []
    for weights in weights_list:
        weights_name = getFileName(weights)

        # print('weights_name: ',weights)

        meta_file,cfg_file = getMetaCfgName(weights)
        # meta = dn.load_meta(meta_file)
        # net = dn.load_net(cfg_file,bytes(weights,'utf-8'),0)

        # 选择对应的dn
        meta = dn.load_meta(meta_file)
        net = dn.load_net(cfg_file,bytes(weights,'utf-8'),0)

        object_type = [meta.names[i].decode('utf-8').strip() for i in range(meta.classes)]

        result_path = os.path.join(result_dir,weights_name)
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        # detect result and save to text
        timeall = 0
        for j,image_path in enumerate(image_list):
            print('detect: '+str(j+1)+'/'+str(len(image_list)))
            label_path = imagePath2labelPath(image_path)
            image_name = getFileName(image_path)
            det_save_path = os.path.join(result_path,image_name+'.txt')
            # det = dn.detect_ext(net, meta, bytes(image_path,'utf-8'),thresh)

            # 选择对应的dn
            det,time1 = dn.detect_ext(net, meta, bytes(image_path,'utf-8'),thresh)
            timeall = timeall + time1;

            # save detection result to text
            saveDetRes(det,det_save_path,object_type)
            time.sleep(0.001)
        print('xxxxxxxxxxx', 'FPS, ',len(image_list)/timeall)
        # dn.free_net(net)

        # campare label and detection result
        for i,objtype in enumerate(object_type):

            # if objtype != 'fr':
            #     continue
            total_label = 0
            total_detect = 0
            total_corr = 0
            total_iou = 0
            cmp_result = []
            det_ = []
            annopath = []

            detall = [['name','obj_type', 'score',0,0,0,0]] # 此处为xywh(中心)，应该变为xmin,ymin,xmax,ymax

            imagesetfile = []
            for j,image_path in enumerate(image_list):
                label_path = imagePath2labelPath(image_path)
                image_name = getFileName(image_path)
                imagesetfile.append(image_name)
                img_save_path = os.path.join(result_path,image_name+'.jpg')
                det_save_path = os.path.join(result_path,image_name+'.txt')

                # detpath.append(det_save_path)
                annopath.append(label_path)
                # print(img_save_path)
                label = []
                if os.path.exists(label_path):
                    label = LoadLabel(label_path,object_type)

                # save detection result to text
                det = readDetRes(det_save_path)
                for d in det:
                    if d[0] > len(object_type)-1:
                        d[0] = ' '
                        continue 
                    d[0] = object_type[d[0]]

                for d in det:
                    xmin = float(copy.deepcopy(d[2])) - float(copy.deepcopy(d[4]))/2.0
                    ymin = float(copy.deepcopy(d[3])) - float(copy.deepcopy(d[5]))/2.0
                    xmax = float(copy.deepcopy(d[2])) + float(copy.deepcopy(d[4]))/2.0
                    ymax = float(copy.deepcopy(d[3])) + float(copy.deepcopy(d[5]))/2.0
                    # 该文件格式：imagename1 type confidence xmin ymin xmax ymax
                    d_ = [image_name, d[0], d[1], xmin, ymin, xmax, ymax]
                    det_.append(d_)

                if len(det_) != 0:
                    detall = numpy.vstack((detall, det_))
                det_=[]

                if i > 0:
                    image_path = img_save_path
                # print(j,image_path)
                img = cv2.imread(image_path)
                if img is None:
                    print("load image error&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    continue

                cmp_res = cdl.CmpData(objtype,det,label,thresh,iou_thresh,img)

                cmp_res.update({'image_name':image_name})
                total_corr += cmp_res['correct']
                total_iou += cmp_res['avg_iou']*cmp_res['label_num']

                cmp_result.append(cmp_res)
                print("%s: %d/%d  label: %d   detect: %d   correct: %d   recall: %f   avg_iou: %f   accuracy: %f   precision: %f\n" % \
                    (str(objtype),j+1,image_num,cmp_res['label_num'],cmp_res['detect_num'],\
                        cmp_res['correct'],cmp_res['recall'],cmp_res['avg_iou'],\
                            cmp_res['accuracy'],cmp_res['precision']))
                total_label += cmp_res['label_num']
                total_detect += cmp_res['detect_num']
                cv2.imwrite(img_save_path,img)
                img = []
                time.sleep(0.001)

            # 求出AP值
            ap=0
            # detall = numpy.delete(detall, 0, axis = 0)
            # det_objtype = [obj for obj in detall if obj[1] == objtype]
            # if len(det_objtype) == 0:
            #     ap = 0
            # else:
            #     ap = voc_eval(det_objtype, annopath, imagesetfile, objtype, iou_thresh)
            # detall=[]

            #数据集分析结果
            avg_recall = 0
            if total_label > 0:
                avg_recall = total_corr/float(total_label)
            avg_iou = 0
            if total_iou > 0:
                avg_iou = total_iou/total_label
            avg_acc = 0
            if total_label+total_detect-total_corr > 0:
                avg_acc = float(total_corr)/(total_label+total_detect-total_corr)
            avg_precision = 0
            if total_detect > 0:
                avg_precision = float(total_corr)/total_detect
            total_result = [total_label,total_detect,total_corr,avg_recall,avg_iou,avg_acc,avg_precision]
            cdl.ExportAnaRes(objtype,cmp_result,total_result,image_path,result_path)
            print("total_label: %d   total_detect: %d   total_corr: %d   recall: %f   average iou: %f   accuracy: %f   precision: %f ap: %f\n" % \
                (total_result[0],total_result[1],total_result[2],total_result[3],total_result[4],total_result[5],total_result[6],ap))
            
            result.append([weights_name]+[objtype]+total_result+[float(ap)])
        cdl.ExportAnaResAll(result, result_dir)
        time.sleep(0.001)

if __name__ == "__main__":
    
    dn.set_gpu(4)
    weights_list_file = "/users/duanyou/c5/v4_all_train_coco/weights.txt"

    # COCO test
    # data_path = "/users/duanyou/c5/v4_all_train_coco"
    # image_list_file = os.path.join(data_path,"5k.txt")
    # result_dir = os.path.join("/users/duanyou/c5/results_v4_coco/results_all/")
    # batch_analysis(weights_list_file,image_list_file,0.20,0.45,result_dir)

    # all_test
    # data_path = "/users/duanyou/c5/all_pretrain"
    # image_list_file = os.path.join(data_path,"test.txt")
    # result_dir = os.path.join("/users/duanyou/c5/results_v4_coco/results_all/")
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    # batch_analysis(weights_list_file,image_list_file,0.20,0.45,result_dir)

    # changsha_test
    # data_path = "/users/duanyou/c5/changsha"
    # image_list_file = os.path.join(data_path,"test.txt")
    # result_dir = os.path.join("/users/duanyou/c5/results_v4_coco/results_changsha/")
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    # batch_analysis(weights_list_file,image_list_file,0.20,0.45,result_dir)

    # hezhoupucheng_test
    # data_path = "/users/duanyou/c5/hezhoupucheng"
    # image_list_file = os.path.join(data_path,"test.txt")
    # result_dir = os.path.join("/users/duanyou/c5/results_v4_coco/results_hezhoupucheng/")
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    # batch_analysis(weights_list_file,image_list_file,0.20,0.45,result_dir)

    # puer_test
    # data_path = "/users/duanyou/c5/puer"
    # image_list_file = os.path.join(data_path,"test.txt")
    # result_dir = os.path.join("/users/duanyou/c5/results_v4_coco/results_puer/")
    # if not os.path.exists(result_dir):
    #     os.mkdir(result_dir)
    # batch_analysis(weights_list_file,image_list_file,0.20,0.45,result_dir)

    # yancheng_test
    data_path = "/users/duanyou/c5/yancheng"
    image_list_file = os.path.join(data_path,"test.txt")
    result_dir = os.path.join("/users/duanyou/c5/results_v4_coco/results_yancheng/")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    batch_analysis(weights_list_file,image_list_file,0.20,0.45,result_dir)
    