# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import numpy as np


def convert(size, box):  # box=xmin,ymin,xmax,ymax
    dw = 1. / size[0]
    dh = 1. / size[1]
    xmin = box[0] * dw
    ymin = box[1] * dh
    xmax = box[2] * dw
    ymax = box[3] * dh
    return (xmin, ymin, xmax, ymax)


def parse_rec(filename):  # 读取标注的xml文件
    """ Parse a PASCAL VOC xml file """
    in_file = open(filename)
    xml_info = in_file.read()
    try:
        root = ET.fromstring(xml_info)
    except(Exception, e):
        print("Error: cannot parse file")
    objects = []
    if root.find('markNode') != None:
        obj = root.find('markNode').find('object')
        if obj != None:
            w = int(root.find('width').text)
            h = int(root.find('height').text)
            for obj in root.iter('object'):
                if 'non_interest' in str(obj.find('targettype').text):
                    continue
                obj_struct = {}
                if obj.find('targettype').text == 'car_rear' or obj.find('targettype').text == 'car_front':
                    obj_struct['name'] = 'fr'
                else:
                    obj_struct['name'] = obj.find('targettype').text
                obj_struct['pose'] = 0  # obj.find('pose').text
                obj_struct['truncated'] = 0  # int(obj.find('truncated').text)
                obj_struct['difficult'] = 0  # int(obj.find('difficult').text)
                # bbox = obj.find('bndbox')
                b = [float(obj.find('bndbox').find('xmin').text),
                     float(obj.find('bndbox').find('ymin').text),
                     float(obj.find('bndbox').find('xmax').text),
                     float(obj.find('bndbox').find('ymax').text)]
                bb = convert((w, h), b)
                if bb is None:
                    continue
                obj_struct['bbox'] = [bb[0], bb[1], bb[2], bb[3]]
                objects.append(obj_struct)
    return objects


def voc_ap(rec, prec):
    # 采用更为精确的逐点积分方法
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5):
    # 主函数，计算当前类别的recall和precision
    # #detpath检测结果txt文件，路径VOCdevkit/results/VOC20xx/Main/<comp_id>_det_test_aeroplane.txt。
    # 该文件格式：imagename1 type confidence xmin ymin xmax ymax  (图像1的第一个结果)
    #           imagename1 type confidence xmin ymin xmax ymax  (图像1的第二个结果)
    #           imagename1 type confidence xmin ymin xmax ymax  (图像2的第一个结果)
    #           ......
    # 每个结果占一行，检测到多少个BBox就有多少行，这里假设有20000个检测结果

    # detpath: Path to detections
    #     detpath.format(classname) should produce the detection results file.
    # annopath: Path to annotations
    #     annopath.format(imagename) should be the xml annotations file. #xml 标注文件。
    # imagesetfile: Text file containing the list of images, one image per line. #数据集划分txt文件，路径VOCdevkit/VOC20xx/ImageSets/Main/test.txt这里假设测试图像1000张，那么该txt文件1000行。
    # classname: Category name (duh) #种类的名字，即类别，假设类别2（一类目标+背景）。
    # cachedir: Directory for caching the annotations #缓存标注的目录路径VOCdevkit/annotation_cache,图像数据只读文件，为了避免每次都要重新读数据集原始数据。
    # [ovthresh]: Overlap threshold (default = 0.5) #重叠的多少大小。
    # [use_07_metric]: Whether to use VOC07's 11 point AP computation
    #     (default False) #是否使用VOC07的AP计算方法，voc07是11个点采样。

    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    imagenames = [x.strip() for x in imagesetfile]

    # parse_rec函数读取当前图像标注文件，返回当前图像标注，存于recs字典（key是图像名，values是gt）
    recs = {}
    for i, imagename in enumerate(imagenames):
        # recs[imagename] = parse_rec(annopath.format(imagename))
        recs[imagename] = parse_rec(annopath[i])

    # extract gt objects for this class #按类别获取标注文件，recall和precision都是针对不同类别而言的，AP也是对各个类别分别算的。
    class_recs = {}  # 当前类别的标注
    npos = 0  # npos标记的目标数量
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]  # 过滤，只保留recs中指定类别的项，存为R。
        bbox = np.array([x['bbox'] for x in R])  # 抽取bbox
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)  # 如果数据集没有difficult,所有项都是0.

        det = [False] * len(R)  # len(R)就是当前类别的gt目标个数，det表示是否检测到，初始化为false。
        npos = npos + sum(~difficult)  # 自增，非difficult样本数量，如果数据集没有difficult，npos数量就是gt数量。
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets 读取检测结果
    splitlines = detpath  # 该文件格式：imagename1 type confidence xmin ymin xmax ymax
    # splitlines = [x.strip().split(' ') for x in detpath]  # 假设检测结果有20000个，则splitlines长度20000
    image_ids = [x[0] for x in splitlines]  # 检测结果中的图像名，image_ids长度20000，但实际图像只有1000张，因为一张图像上可以有多个目标检测结果
    confidence = np.array([float(x[2]) for x in splitlines])  # 检测结果置信度
    BB = np.array([[float(z) for z in x[3:]] for x in splitlines])  # 变为浮点型的bbox。

    npos = len(image_ids)

    # sort by confidence 将20000各检测结果按置信度排序
    sorted_ind = np.argsort(-confidence)  # 对confidence的index根据值大小进行降序排列。
    sorted_scores = np.sort(-confidence)  # 降序排列。
    BB = BB[sorted_ind, :]  # 重排bbox，由大概率到小概率。
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)  # 注意这里是20000，不是1000
    tp = np.zeros(nd)  # true positive，长度20000
    fp = np.zeros(nd)  # false positive，长度20000
    for d in range(nd):  # 遍历所有检测结果，因为已经排序，所以这里是从置信度最高到最低遍历
        R = class_recs[image_ids[d]]  # 当前检测结果所在图像的所有同类别gt
        bb = BB[d, :].astype(float)  # 当前检测结果bbox坐标
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)  # 当前检测结果所在图像的所有同类别gt的bbox坐标

        if BBGT.size > 0:
            # compute overlaps 计算当前检测结果，与该检测结果所在图像的标注重合率，一对多用到python的broadcast机制
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)  # 最大重合率
            jmax = np.argmax(overlaps)  # 最大重合率对应的gt,返回最大索引数
            # print('overlaps',overlaps,'ovmax',ovmax,'jmax ',jmax)

        if ovmax > ovthresh:  # 如果当前检测结果与真实标注最大重合率满足阈值
            # if not R['difficult'][jmax]:
            if not R['det'][jmax]:
                tp[d] = 1.  # 正检数目+1
                R['det'][jmax] = True  # 该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
            else:  # 相反，认为检测到一个虚警
                fp[d] = 1.
        else:  # 不满足阈值，肯定是虚警
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)  # 积分图，在当前节点前的虚警数量，fp长度
    tp = np.cumsum(tp)  # 积分图，在当前节点前的正检数量
    rec = tp / float(npos)  # 召回率，长度20000，从0到1
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth 准确率，长度20000，长度20000，从1到0
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return ap
