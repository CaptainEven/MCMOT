import os
import cv2
import xlwt


def overlap(x1, w1, x2, w2):
    l1 = x1 - w1 / 2.
    l2 = x2 - w2 / 2.
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1 / 2.
    r2 = x2 + w2 / 2.
    right = r1 if r1 < r2 else r2
    return right - left


def box_intersection(box1, box2):
    w = overlap(box1[0], box1[2], box2[0], box2[2])
    h = overlap(box1[1], box1[3], box2[1], box2[3])
    if w < 0 or h < 0:
        return 0
    area = w * h
    return area


def box_union(box1, box2):
    i = box_intersection(box1, box2)
    u = box1[2] * box1[3] + box2[2] * box2[3] - i
    return u


def box_iou(box1, box2):
    return box_intersection(box1, box2) / box_union(box1, box2)


def box_to_rect(box, width, height):
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    left = (x - w / 2.) * width
    top = (y - h / 2.) * height
    right = (x + w / 2.) * width
    bottom = (y + h / 2.) * height
    return [int(left), int(top), int(right), int(bottom)]


# 比较每张图片的检测结果和标记数据
def CmpData(cmp_type, detect_objs, label_objs, thresh, iou_thresh, img):
    # img = cv2.imread("%s/%s.jpg" % (image_path,file_name))

    df = [False for n in range(0, len(detect_objs))]
    correct = 0
    iou = 0
    label_num = 0
    for lobj in label_objs:
        if lobj[0] != cmp_type:
            continue
        label_num += 1
        box1 = [lobj[1], lobj[2], lobj[3], lobj[4]]
        rect1 = box_to_rect(box1, img.shape[1], img.shape[0])
        best_iou = 0
        rect2 = []
        best_no = -1
        for dno, dobj in enumerate(detect_objs):
            if lobj[0] != dobj[0]:
                continue
            box2 = [dobj[2], dobj[3], dobj[4], dobj[5]]
            biou = box_iou(box1, box2)
            if dobj[1] > thresh and biou > best_iou:
                best_no = dno
                best_iou = biou
                rect2 = box_to_rect(box2, img.shape[1], img.shape[0])
        iou += best_iou
        # if best_iou > iou_thresh:
        if best_iou > iou_thresh and not df[best_no]:  #### 若df[best_no]已经是true了，则证明这个检测结果没有匹配的GT，且置信度大于thresh，则算虚警
            correct += 1
            df[best_no] = True  # df相当于该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
            # cv2.rectangle(img,(rect1[0],rect1[1]),(rect1[2],rect1[3]),(0,255,0),3)# 绿色 label
            if cmp_type == 'car':
                cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (255, 0, 0), 3)
                txt = cmp_type + ':' + str(round(detect_objs[best_no][1], 2))
                cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (255, 0, 0), 2)
            elif cmp_type == 'bicycle':
                cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (255, 255, 0), 3)
                txt = cmp_type + ':' + str(round(detect_objs[best_no][1], 2))
                cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (255, 255, 0), 2)
            elif cmp_type == 'person':
                cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (0, 255, 255), 3)
                txt = cmp_type + ':' + str(round(detect_objs[best_no][1], 2))
                cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (0, 255, 255), 2)
            elif cmp_type == 'cyclist':
                cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (0, 255, 0), 3)
                txt = cmp_type + ':' + str(round(detect_objs[best_no][1], 2))
                cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (0, 255, 0), 2)
            elif cmp_type == 'tricycle':
                cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (0, 0, 255), 3)
                txt = cmp_type + ':' + str(round(detect_objs[best_no][1], 2))
                cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (0, 0, 255), 2)
            elif cmp_type == 'fr':
                cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (255, 0, 255), 3)
                txt = 'fr' + ':' + str(round(detect_objs[best_no][1], 2))
                cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (255, 0, 255), 2)
        # else:
        #     cv2.rectangle(img,(rect1[0],rect1[1]),(rect1[2],rect1[3]),(0,255,255),3) # 黄色，未检测到的GT

    detect_num = 0
    for i, dobj in enumerate(detect_objs):
        if dobj[0] != cmp_type:
            continue
        if dobj[1] > thresh:
            detect_num += 1
        box2 = [dobj[2], dobj[3], dobj[4], dobj[5]]
        if not df[i]:  # 如果df[i]=False，则表明这个检测结果没有匹配的GT，且置信度大于thresh，则算虚警，相当于R['det'][jmax]
            if dobj[1] > thresh:
                rect2 = box_to_rect(box2, img.shape[1], img.shape[0])
                # cv2.rectangle(img,(rect2[0],rect2[1]),(rect2[2],rect2[3]),(255,0,0),3) # 红色 虚警
                # if cmp_type == 'fr':
                #     cmp_type1 = 'shangfan'
                # else:
                #     cmp_type1 = cmp_type
                # txt = cmp_type1+':'+str(round(dobj[1],2))
                # cv2.putText(img,txt,(rect2[0],rect2[1]), 0, 1, (255,0,0),2)
                if cmp_type == 'car':
                    cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (255, 0, 0), 3)
                    txt = cmp_type + ':' + str(round(dobj[1], 2))
                    cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (255, 0, 0), 2)
                elif cmp_type == 'bicycle':
                    cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (255, 255, 0), 3)
                    txt = cmp_type + ':' + str(round(dobj[1], 2))
                    cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (255, 255, 0), 2)
                elif cmp_type == 'person':
                    cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (0, 255, 255), 3)
                    txt = cmp_type + ':' + str(round(dobj[1], 2))
                    cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (0, 255, 255), 2)
                elif cmp_type == 'cyclist':
                    cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (0, 255, 0), 3)
                    txt = cmp_type + ':' + str(round(dobj[1], 2))
                    cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (0, 255, 0), 2)
                elif cmp_type == 'tricycle':
                    cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (0, 0, 255), 3)
                    txt = cmp_type + ':' + str(round(dobj[1], 2))
                    cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (0, 0, 255), 2)
                elif cmp_type == 'fr':
                    cv2.rectangle(img, (rect2[0], rect2[1]), (rect2[2], rect2[3]), (255, 0, 255), 3)
                    txt = 'fr' + ':' + str(round(dobj[1], 2))
                    cv2.putText(img, txt, (rect2[0], rect2[1]), 0, 1, (255, 0, 255), 2)

    # cv2.imwrite("%s/show_result/%s_r.jpg" % (result_path,file_name),img)

    tp = correct
    fp = detect_num - tp
    tn = 0
    fn = label_num - tp
    avg_iou = 0
    recall = 0
    accuracy = 0
    precision = 0
    if 0 == label_num:
        avg_iou = 0
        recall = 1
        accuracy = 1 if detect_num == 0 else 0
        precision = 1 if detect_num == 0 else 0
    else:
        avg_iou = iou / label_num
        recall = correct / float(label_num)
        accuracy = correct / float(tp + fn + fp + tn)
        corr = (correct if correct < detect_num else detect_num)  # 检测正确数大于检测结果数的情况，即同一个目标多次标记
        precision = 0 if detect_num == 0 else corr / float(detect_num)

    cmp_res = {'label_num': label_num, 'detect_num': detect_num, 'correct': correct, \
               'recall': recall, 'avg_iou': avg_iou, 'accuracy': accuracy, 'precision': precision}

    return cmp_res


# 输出分析结果到excel文件中
def ExportAnaRes(objtype, res1, total_result, image_path, result_path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
    row0 = [u'图片名', u'标注目标', u'检测目标', u'检测正确', u'recall', u'iou', u'accuracy', u'precision']
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i])

    for r in range(0, len(res1)):
        sheet1.write(r + 1, 0, res1[r]['image_name'])
        sheet1.write(r + 1, 1, res1[r]['label_num'])
        sheet1.write(r + 1, 2, res1[r]['detect_num'])
        sheet1.write(r + 1, 3, res1[r]['correct'])
        sheet1.write(r + 1, 4, res1[r]['recall'])
        sheet1.write(r + 1, 5, res1[r]['avg_iou'])
        sheet1.write(r + 1, 6, res1[r]['accuracy'])
        sheet1.write(r + 1, 7, res1[r]['precision'])

    row_end = [u'total', total_result[0], total_result[1], total_result[2], total_result[3], \
               total_result[4], total_result[5], total_result[6]]
    for i in range(0, len(row_end)):
        sheet1.write(len(res1) + 2, i, row_end[i])

    save_name = "AnalyseResult_%s.xls" % (objtype)
    save_path = os.path.join(result_path, save_name)
    f.save(save_path)


def ExportAnaResAll(results, result_path):
    f = xlwt.Workbook()  # 创建工作簿
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)
    row0 = [u'模型', u'目标类型', u'标注目标', u'检测目标', u'检测正确', u'recall', u'iou', u'accuracy', u'precision', u'AP']
    for i in range(0, len(row0)):
        sheet1.write(0, i, row0[i])
    for r in range(len(results)):
        total_result = results[r]
        for i in range(0, len(results[r])):
            sheet1.write(r + 1, i, results[r][i])

    save_path = os.path.join(result_path, 'AnalyseResultAll.xls')
    f.save(save_path)
