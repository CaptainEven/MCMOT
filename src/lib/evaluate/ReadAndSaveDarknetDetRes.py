import os


# import darknet as dn

def read_det_res(res_path):
    fr = open(res_path, 'r')
    if fr is None:
        return -1
    cn = 0
    num = 0
    detect_objs = []
    for line in fr.readlines():  # 依次读取每行
        line = line.strip()  # 去掉每行头尾空白
        if cn == 0:
            tmp, num = [str(i) for i in line.split("=")]
            # print("object num: ", int(num))
        else:
            obj = [float(i) for i in line.split()]
            obj[0] = int(obj[0])
            detect_objs.append(obj)
            # print(obj)
        cn += 1

    return detect_objs


def save_det_res(det, det_save_path, cls_names):
    """
    :param det:
    :param det_save_path:
    :param cls_names:
    :return:
    """
    res = 0
    f = open(det_save_path, 'w')
    if f is None:
        res = -1
        return res

    f.write('class prob x y w h total=' + str(len(det)) + '\n')
    for d in det:
        if d[0] not in cls_names:
            res = -2
            continue
        obj_cls = cls_names.index(d[0])
        f.write('%d %f %f %f %f %f\n' % (obj_cls, d[1], d[2], d[3], d[4], d[5]))
        # print(obj_cls,d[2],d[3],d[4],d[5])

    return res


if __name__ == "__main__":
    # detect
    print('done')
    # net = dn.load_net(b"/users/maqiao/mq/Data_checked/multiClass/backup_yolov3-spp/multiClass_yolov3-spp_test.cfg", 
    #                 b"/users/maqiao/mq/Data_checked/multiClass/backup_yolov3-spp/multiClass_yolov3-spp_60000.weights", 0)
    # meta = dn.load_meta(b"/users/maqiao/mq/Data_checked/multiClass/backup_c5/multiClass.data")
    # r = dn.detect_ext(net, meta, b"/users/maqiao/mq/Data_checked/multiClass/multiClass0320/JPEGImages_ori/000000.jpg")
    # dn.free_net(net)
    # print(meta.classes)
    # for c in range(meta.classes):
    #     print(meta.names[c])
    # print(r)

    # # save detection result to text
    # cls_names = [meta.names[i].decode('utf-8').strip() for i in range(meta.classes)]
    # saveDetRes(r, 'result.txt', cls_names)

    # # read detection result
    # objs = readDetRes('result.txt')
    # print(objs)
