import os
import darknet as dn
import cv2
import shutil
import numpy as np

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print("copy %s -> %s"%( srcfile,dstfile))

def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print("move %s -> %s"%( srcfile,dstfile))

def listdir(path):
    list_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_name += listdir(file_path)
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

# 计算前后帧之间的多个检测框间的iou
def batch_iou(boxes1, boxes2, width, height):
    img1 = np.zeros((height,width), dtype=np.int)
    for b in boxes1:
        x1 = int(b[0]*width)
        x2 = x1+int(b[2]*width)
        y1 = int(b[1]*height)
        y2 = y1+int(b[3]*height)
        img1[y1:y2,x1:x2] = 1
    img2 = np.zeros((height,width), dtype=np.int)
    for b in boxes2:
        x1 = int(b[0]*width)
        x2 = x1+int(b[2]*width)
        y1 = int(b[1]*height)
        y2 = y1+int(b[3]*height)
        img2[y1:y2,x1:x2] = 1
    img = img1 + img2
    union = np.where(img>0)
    inter = np.where(img>1)
    iou = float(len(inter[0]))/len(union[0])
    return iou
    
def batch_analysis(meta_file,cfg_file,wgt_file,thresh,nms,src_path,dst_path):

    image_list = listdir(src_path)
    image_list.sort()
    image_num = len(image_list)
    meta = dn.load_meta(meta_file)
    object_type = [meta.names[i].decode('utf-8').strip() for i in range(meta.classes)]
    net = dn.load_net(cfg_file,wgt_file,0)
    move_count = 0
    boxes_last = []
    
    for j,image_path in enumerate(image_list):
    
        print(str(j)+'/'+str(image_num)+"  moved: "+str(move_count))
        # print(image_path)
        
        try:
            img = cv2.imread(image_path)
        except:
            print('can not read image******************************************')
            continue
        h,w = img.shape[:2]
        image_name = getFileName(image_path)
        print("image_name", image_name)
        image_name = image_name.replace('(','1_')
        image_name = image_name.replace(')','_1')
        img_save_path = os.path.join(dst_path,image_name+'.jpg')
        # print(img_save_path)
        det = dn.detect_ext(net, meta, bytes(image_path,'utf-8'),thresh)
        boxes = []
        is_move_file = False
        
        if j%20 == 0:   #20数值越大 比对iou的间隔越大
            is_move_file = True
            
        for d in det:
            # try:
            #     img = cv2.imread(image_path)
            # except:
            #     print('can not read image******************************************')
            #     continue
            # h,w = img.shape[:2]
            print("d",d)
            boxes.append(d[2:])
            bw = d[4]*w
            bh = d[5]*h
#            if bw < 20 or bh < 20:
#                print("bw or bh is less than 20")
#                continue
#            obj_type = d[0]
#            if obj_type == 'tricycle':
#                print("tricycle ************************************************")
#                is_move_file = True
#                break
#            elif obj_type == 'car':
#                if bw*bh/(w*h) > 0.25:
#                    print("big car ....................................................")
#                    is_move_file = True
#                    break
        if boxes_last != [] and boxes != []:
            iou = batch_iou(boxes_last,boxes,w,h)
            # print('iou: '+str(iou))
            if iou > 0.6:
                print('batch iou: '+str(iou))
                is_move_file = False
                print("iou^^^^^^^^^^^^^^^^^^^^^^^^^")
                # continue
        if is_move_file:
            move_count += 1
            if not os.path.exists(img_save_path):
                mymovefile(image_path,img_save_path)
            boxes_last = boxes
    dn.free_net(net)


if __name__ == "__main__":
    # dn.set_gpu(3)
    src_path = "/mnt/diskc/zhoukai/puer0605/" # 原始的图片目录
    dst_path = "/mnt/diskc/zhoukai/puer0605/puer_jingjian" # 过滤后的图片目录
    cfg_file = b"/users/duanyou/c5/v4_all_train/yolov4_test.cfg"
    wgt_file = b"/users/duanyou/c5/v4_all_train/yolov4_5000.weights"
    meta_file = b"/users/duanyou/c5/v4_all_train/multiClass.data"
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    batch_analysis(meta_file,cfg_file,wgt_file,0.2,0.45,src_path,dst_path)
    