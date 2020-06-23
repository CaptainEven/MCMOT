import os
import darknet as dn
import cv2
import shutil
from lxml import etree, objectify
import os,glob
import xml.etree.ElementTree as ET

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

def listdir(path, ftype):
    list_name = []
    for f in os.listdir(path):
        if os.path.splitext(f)[-1] != ftype:
            continue
        file_path = os.path.join(path, f)
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
    file_name = file_name.replace('.jpg', '').replace('.png', '')
    # p = file_name.split('.')
    # name = ''
    # for i in range(len(p)-1):
    #     name += p[i]
    # file_name = p[]
    return file_name

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

def writeXml(xmlfile, imgW, imgH, img_name, det_result):
    E = objectify.ElementMaker(annotate=False)
    anno_dataroot = E.dataroot(
        E.folder(''),
        E.filename(img_name),
        E.createdata(''),
        E.modifydata(''),
        E.width(imgW),
        E.height(imgH),
        E.DayNight(''),
        E.weather(''),
        E.Marker('Alg'),
        E.location(''),
        E.imageinfo(''),
        E.source(''),
        E.database('')
    )

    E_markNode = objectify.ElementMaker(annotate=False)
    anno_markNode = E_markNode.markNode()

    for i,obj in enumerate(det_result[0]):
        # print('det_result: ', det_result)
        # print('obj: ', obj)
        targettype = obj[0]
        x = obj[2]*imgW
        y = obj[3]*imgH
        w = obj[4]*imgW
        h = obj[5]*imgH
        xmin = (int)(x - w/2)
        ymin = (int)(y - h/2)
        xmax = (int)(x + w/2)
        ymax = (int)(y + h/2)
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > imgW - 1:
            xmax = imgW - 1
        if ymax > imgH - 1:
            ymax = imgH - 1
        if xmax - xmin <= 65:
            print(obj[0],x,y,w,h)
            print('obj width less than 10')
            continue
        if ymax - ymin <= 65:
            print(obj[0],x,y,w,h)
            print('obj height less than 10')
            continue
        cartype = ''
        # if targettype == 'car_front':
        #     continue
        if targettype == 'fr':
            targettype = 'car_front'

        if targettype == 'car' or targettype == 'car_front':
            cartype = 'saloon_car'
        
        E_object = objectify.ElementMaker(annotate=False)
        anno_object = E_object.object(
            E_object.index(i+1),
            E_object.targettype(targettype),
            E_object.cartype(cartype),
            E_object.cartypechild(),
            E_object.pose(),
            E_object.truncated(),
            E_object.difficult(),
            E_object.remark()
        )

        E_bndbox = objectify.ElementMaker(annotate=False)
        anno_bndbox = E_bndbox.bndbox(
            E_bndbox.xmin(xmin),
            E_bndbox.ymin(ymin),
            E_bndbox.xmax(xmax),
            E_bndbox.ymax(ymax)
        )
        anno_object.append(anno_bndbox)
        anno_markNode.append(anno_object)
    anno_dataroot.append(anno_markNode)

    etree.ElementTree(anno_dataroot).write(xmlfile, encoding='utf-8', xml_declaration=True)


def batch_analysis(meta_file,cfg_file,wgt_file,meta_file_fr,cfg_file_fr,wgt_file_fr,
                    thresh,nms,img_path,xml_path):
    image_list = listdir(img_path,'.jpg')
    image_num = len(image_list)
    meta = dn.load_meta(meta_file)
    net = dn.load_net(cfg_file,wgt_file,0)
    # meta_fr = dn.load_meta(meta_file_fr)
    # net_fr = dn.load_net(cfg_file_fr,wgt_file_fr,0)
    move_count = 0
    for j,image_path in enumerate(image_list):
        print(str(j)+'/'+str(image_num)+"  "+image_path)
        image_name = getFileName(image_path)
        img_save_path = os.path.join(img_path,image_name+'.jpg')
        xml_save_path = os.path.join(xml_path,image_name+'.xml')
        # if os.path.exists(xml_save_path):
        #     continue
        # print(img_save_path)
        det = dn.detect_ext(net, meta, bytes(image_path,'utf-8'),thresh)
        # det_fr = dn.detect_ext(net_fr, meta_fr, bytes(image_path,'utf-8'),thresh)
        img = cv2.imread(image_path)
        if img is None:
            print('Can not open image')
            continue
        h,w,c = img.shape
        writeXml(xml_save_path,w,h,image_name,det)
    dn.free_net(net)

def batch_analysis_c6(meta_file,cfg_file,wgt_file,thresh,nms,img_path,xml_path):
    image_list = listdir(img_path,'.jpg')
    image_num = len(image_list)
    meta = dn.load_meta(meta_file)
    net = dn.load_net(cfg_file,wgt_file,0)
    # meta_fr = dn.load_meta(meta_file_fr)
    # net_fr = dn.load_net(cfg_file_fr,wgt_file_fr,0)
    move_count = 0
    for j,image_path in enumerate(image_list):
        print(str(j)+'/'+str(image_num)+"  "+image_path)
        image_name = getFileName(image_path)
        img_save_path = os.path.join(img_path,image_name+'.jpg')
        xml_save_path = os.path.join(xml_path,image_name+'.xml')
        # if os.path.exists(xml_save_path):
        #     continue
        # print(img_save_path)
        det = dn.detect_ext(net, meta, bytes(image_path,'utf-8'),thresh)
        # det_fr = dn.detect_ext(net_fr, meta_fr, bytes(image_path,'utf-8'),thresh)
        img = cv2.imread(image_path)
        if img is None:
            print('Can not open image')
            continue
        h,w,c = img.shape
        writeXml(xml_save_path,w,h,image_name,det)
    dn.free_net(net)

if __name__ == "__main__":
    dn.set_gpu(5)
    # img_path = "/mnt/diskc/maqiao/data/20191104/JPEGImages/JPEGImages"

    # 11.25，需要夏燎安排人标注的
    # img_path = '/mnt/diskc/maqiao/data/20191122'
    # img_path = '/mnt/diskc/maqiao/data/yc20191101~20191119/train'
    img_path = '/mnt/diskd/Data_all/SCSN0002-7-12-15'
    # img_path = '/mnt/diskd/Data_all/待标注数据20200616'
    # img_path = '/users/duanyou/backup_c5/test_1/JPEGImages'
    # img_path = '/mnt/diskb/duanyou/需要标注的数据/shangfang_20200605'
    # img_path = '/users/duanyou/backup_c5/test_4/train'
    # img_path = '/users/duanyou/backup_c5/test_2/1230标注'
    # img_path = '/mnt/diskd/Data_all/多目标类型/需要标注的垂停20191217-大连-蒲城-盐城-长沙/train'


    xml_path = img_path
    # if not os.path.exists(xml_path):
    #     os.mkdir(xml_path)
    # xml_path_fr = os.path.join(img_path,'FR_xml')
    # if not os.path.exists(xml_path_fr):
    #     os.mkdir(xml_path_fr)

    # ## multiClass_c5
    # cfg_file = b"/users/maqiao/mq/Data_checked/multiClass/backup_yolov3-spp/multiClass_yolov3-spp_test.cfg"
    # wgt_file = b"/users/maqiao/mq/Data_checked/multiClass/backup_yolov3-spp/multiClass_yolov3-spp_145000.weights"
    # meta_file = b"/users/maqiao/mq/Data_checked/multiClass/backup_c5/multiClass.data"

    # # ## FR
    # cfg_file_fr = b"models/FR/tiny-yolo-voc-decode.cfg"
    # wgt_file_fr = b"models/FR/tiny_yolo_voc_FR_final.weights"
    # meta_file_fr = b"models/FR/FR.data"

    # ## hzpc
    # cfg_file_c6 = b"/users/duanyou/c5/hezhoupucheng/multiClass_test.cfg"
    # wgt_file_c6 = b"/users/duanyou/c5/hezhoupucheng/multiClass_1084000_20200526.weights"
    # meta_file_c6 = b"/users/duanyou/c5/hezhoupucheng/multiClass.data"

    # ## multiClass_c6， 直接用c6的模型跑全部结果【c6 垂停】
    # cfg_file_c6 = b"/users/duanyou/backup_c6/experiments/c6_chuiting/multiClass_c6_test.cfg"
    # wgt_file_c6 = b"/users/duanyou/backup_c6/experiments/c6_chuiting/multiClass_c6_891000_20200310_best.weights"
    # meta_file_c6 = b"/users/duanyou/backup_c6/experiments/c6_chuiting/multiClass_c6.data"

    # new model
    cfg_file_c6 = b"/users/duanyou/c5/v4_all_train/v4all_mish_for_yujiazai/yolov4_test.cfg"
    wgt_file_c6 = b"/users/duanyou/c5/v4_all_train/v4all_mish_for_yujiazai/yolov4_19000.weights"
    meta_file_c6 = b"/users/duanyou/c5/v4_all_train/multiClass.data"

    # batch_analysis(meta_file,cfg_file,wgt_file,meta_file_fr,cfg_file_fr,wgt_file_fr,
    #                 0.25,0.45,img_path,xml_path)
    batch_analysis_c6(meta_file_c6,cfg_file_c6,wgt_file_c6,0.25,0.45,img_path,xml_path)
