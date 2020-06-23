import os
import darknet as dn
import cv2
import time

import cmp_det_label as cdl
from readAndSaveDarknetDetRes import readDetRes,saveDetRes
from readAnnotations import LoadLabel

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

def batch_detection():
    pass

def batch_analysis(weights_list_file, cfg_file, meta_file, image_list_file, thresh, iou_thresh,result_dir):
    image_list = LoadFileList(image_list_file)
    image_num = len(image_list)
    weights_list = LoadFileList(weights_list_file)
    meta = dn.load_meta(meta_file)
    # print(meta.classes)
    object_type = [meta.names[i].decode('utf-8').strip() for i in range(meta.classes)]
    result = []
    for weights in weights_list:
        weights_name = getFileName(weights)
        result_path = os.path.join(result_dir,weights_name)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        net = dn.load_net(cfg_file,bytes(weights,'utf-8'),0)

        # detect result and save to text
        for j,image_path in enumerate(image_list):
            print('detect: '+str(j+1)+'/'+str(len(image_list)))
            label_path = imagePath2labelPath(image_path)
            image_name = getFileName(image_path)
            det_save_path = os.path.join(result_path,image_name+'.txt')
            # print(img_save_path)
            det = dn.detect_ext(net, meta, bytes(image_path,'utf-8'),thresh)
            # save detection result to text
            saveDetRes(det[0],det_save_path,object_type)
            time.sleep(0.001)
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
            # print(objtype)
            for j,image_path in enumerate(image_list):
                label_path = imagePath2labelPath(image_path)
                image_name = getFileName(image_path)
                img_save_path = os.path.join(result_path,image_name+'.jpg')
                det_save_path = os.path.join(result_path,image_name+'.txt')
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
                # print(label)
                # print(det)
                # print('')
                if i > 0:
                    image_path = img_save_path
                # print(j,image_path)
                img = cv2.imread(image_path)
                if img is None:
                    print("load image error&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                # print(img.shape)
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
                time.sleep(0.001)

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
            print("total_label: %d   total_detect: %d   total_corr: %d   recall: %f   average iou: %f   accuracy: %f   precision: %f \n" % \
                (total_result[0],total_result[1],total_result[2],total_result[3],total_result[4],total_result[5],total_result[6]))
            
            result.append([weights_name]+[objtype]+total_result)
        # dn.free_net(net)
        cdl.ExportAnaResAll(result, result_dir)
        time.sleep(0.001)

if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    # net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    # meta = load_meta("cfg/coco.data")
    # r = detect(net, meta, "data/dog.jpg")
    # print r

    dn.set_gpu(4)
    # data_path = "/users/maqiao/mq/Data_checked/car_test/"
    # data_path = "/mnt/diskc/xiaofan/car_test_20190703"
    # data_path = "/mnt/diskb/maqiao/multiClass/test_c6"
    # data_path = "/users/duanyou/backup_c6/test_c6_fr"
    data_path = "/users/duanyou/backup_c6/project_PLM_20200103"
    # data_path = "/mnt/diskb/maqiao/multiClass/multiClass191104/"

    # weights_list_file = "model/multiClass/weights.txt"
    # cfg_file = b"/users/maqiao/mq/Data_checked/multiClass/backup_c5/multiClass_test.cfg"
    # weights_list_file = "model/multiClass/weights_yolov3-spp.txt"
    # cfg_file = b"/users/maqiao/mq/Data_checked/multiClass/backup_yolov3-spp/multiClass_yolov3-spp_test.cfg"

    # weights_list_file = "model/multiClass/weights_no_pool6.txt"
    # cfg_file = b"/users/maqiao/mq/Data_checked/multiClass/backup_tiny_no_pool6/multiClass_v3tiny_no_pool6_c5_test.cfg"
    # meta_file = b"/users/maqiao/mq/Data_checked/multiClass/backup_c5/multiClass.data"
    
    # 测FR一类的效果，对比我的778000效果
    # weights_list_file = "model/multiClass/weights.txt"
    # cfg_file = b"/users/duanyou/backup_c6/FR/tiny-yolo-voc-decode.cfg"
    # meta_file = b"/users/duanyou/backup_c6/FR/FR.data"

    # 测c5的最好效果，500000与514000的结果，对比我的效果
    # weights_list_file = "model/multiClass/weights.txt"
    # cfg_file = b"/users/duanyou/backup_c5/multiClass_test.cfg"
    # meta_file = b"/users/duanyou/backup_c5/multiClass.data"

    # 最normal的测试
    # weights_list_file = "model/multiClass/weights.txt"
    # cfg_file = b"/users/duanyou/backup_c6/multiClass_c6_test.cfg"
    # meta_file = b"/users/duanyou/backup_c6/multiClass_c6.data"

    # PLM项目的测试
    weights_list_file = "/users/duanyou/backup_c6/project_PLM_20200103/weights.txt"
    cfg_file = b"/users/duanyou/backup_c6/project_PLM_20200103/plm_yolov4_tiny_up1_test.cfg"
    meta_file = b"/users/duanyou/backup_c6/project_PLM_20200103/multiClass_c6.data"

    # weights_list_file = "model/multiClass/weights_yolov3-spp_c6.txt"
    # cfg_file = b"/mnt/diskc/maqiao/Data_checked/multiClass_c6/backup_yolov3-spp_c6_with_fr/multiClass_yolov3-spp_c6_test.cfg"
    # weights_list_file = "model/multiClass/weights_c6.txt"
    # cfg_file = b"/mnt/diskc/maqiao/Data_checked/multiClass_c6/backup_c6/multiClass_c6_test.cfg"
    # meta_file = b"/mnt/diskc/maqiao/Data_checked/multiClass_c6/backup_c6/multiClass_c6.data"

    # weights_list_file = "model/multiClass/weights_no_pool6_c6.txt"
    # cfg_file = b"/mnt/diskc/maqiao/Data_checked/multiClass_c6/backup_no_pool6_c6/multiClass_v3tiny_no_pool6_c6_test.cfg"
    # meta_file = b"/mnt/diskc/maqiao/Data_checked/multiClass_c6/backup_no_pool6_c6/multiClass_v3tiny_no_pool6_c6.data"

    # weights_list_file = 'model/multiClass/fr.txt'
    # cfg_file = b"/users/maqiao/mq/DataPreprocess/models/FR/tiny-yolo-voc-decode.cfg"
    # meta_file = b"/users/maqiao/mq/DataPreprocess/models/FR/FR.data"

    # data_path = "/mnt/diskc/maqiao/Data_checked/tw_motor/test"
    # weights_list_file = "model/tw_motor/weights_tw_motor.txt"
    # cfg_file = b"/mnt/diskc/maqiao/Data_checked/tw_motor/backup/tw_motor_c2_test.cfg"
    # meta_file = b"/mnt/diskc/maqiao/Data_checked/tw_motor/backup/tw_motor_c2.data"
    # weights_list_file = "model/tw_motor/weights_tw_motor_no_pool6.txt"
    # cfg_file = b"/mnt/diskc/maqiao/Data_checked/tw_motor/backup_v3tiny_no_pool6_c3/tw_motor_yolov3tiny_no_pool6_c3_test.cfg"
    # meta_file = b"/mnt/diskc/maqiao/Data_checked/tw_motor/backup_v3tiny_no_pool6_c3/tw_motor_yolov3tiny_no_pool6_c3.data"

    # data_path = "/mnt/diskc/maqiao/Data_checked/car_tw_c2/test"
    # weights_list_file = "model/tw_c2/weights.txt"
    # cfg_file = b"/mnt/diskc/maqiao/Data_checked/car_tw_c2/backup/car_tw_c2_test.cfg"
    # meta_file = b"/mnt/diskc/maqiao/Data_checked/car_tw_c2/backup/car_tw_c2.data"

    image_list_file = os.path.join(data_path,"test.txt")
    result_dir = os.path.join(data_path,"result/")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    batch_analysis(weights_list_file,cfg_file,meta_file,image_list_file,0.20,0.45,result_dir)
    