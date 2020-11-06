from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import logging
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch

my_visible_devs = '0'  # '0, 3'  # 设置可运行GPU编号
os.environ['CUDA_VISIBLE_DEVICES'] = my_visible_devs
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

import cv2
import shutil
import numpy as np
import os.path as osp
import torch.nn.functional as F
from collections import defaultdict
from lib.opts import opts  # import opts
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.log import logger
import lib.datasets.dataset.jde as datasets
from track import eval_seq, eval_imgs_output_dets
from lib.datasets.dataset.jde import letterbox
from lib.models.model import create_model, load_model
from lib.models.decode import mot_decode
from lib.models.utils import _tranpose_and_gather_feat
from lib.tracker.multitracker import map2orig
from lib.tracking_utils.visualization import plot_detects
from lib.utils.utils import select_device

logger.setLevel(logging.INFO)


# find the GPU idx with the largest remaining memory
def FindFreeGPU():
    """
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
    memory_left_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]

    most_free_gpu_idx = np.argmax(memory_left_gpu)
    # print(str(most_free_gpu_idx))
    return int(most_free_gpu_idx)


def run_demo(opt):
    """
    :param opt:
    :return:
    """
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    # clear existing frame results
    frame_res_dir = result_root + '/frame'
    if os.path.isdir(frame_res_dir):
        shutil.rmtree(frame_res_dir)
        os.makedirs(frame_res_dir)
    else:
        os.makedirs(frame_res_dir)

    if opt.input_mode == 'video':
        if opt.id_weight > 0:
            logger.info('Starting tracking...')
        else:
            logger.info('Starting detection...')
        if not os.path.isfile(opt.input_video):
            print('[Err]: invalid input video file.')
            return

        data_loader = datasets.LoadVideo(opt.input_video, opt.img_size)  # load video as input
        f_name = os.path.split(opt.input_video)[-1][:-4]
    elif opt.input_mode == 'image_dir':
        logger.info('Starting detection...')
        data_loader = datasets.LoadImages(opt.input_img, opt.img_size)  # load images as input
        f_name = os.path.split(opt.input_video)[-1]
        opt.id_weight = 0  # only do detection in this mode
    elif opt.input_mode == 'img_path_list_txt':
        logger.info('Starting detection...')
        if not os.path.isfile(opt.input_img):
            print('[Err]: invalid image file path list.')
            return

        opt.id_weight = 0  # only do detection in this mode
        with open(opt.input_img, 'r', encoding='utf-8') as r_h:
            logger.info('Starting detection...')
            paths = [x.strip() for x in r_h.readlines()]
            print('Total {:d} image files.'.format(len(paths)))
            data_loader = datasets.LoadImages(path=paths, img_size=opt.img_size)

    result_file_name = os.path.join(result_root, 'results.txt')
    frame_rate = data_loader.frame_rate
    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')

    # Set device
    # opt.device = device

    # set device
    opt.device = str(FindFreeGPU())
    print('Using gpu: {:s}'.format(opt.device))
    device = select_device(device='cpu' if not torch.cuda.is_available() else opt.device)
    opt.device = device

    try:
        if opt.input_mode == 'video':
            if opt.id_weight > 0:
                eval_seq(opt=opt,
                         data_loader=data_loader,
                         data_type='mot',
                         result_f_name=result_file_name,
                         save_dir=frame_dir,
                         show_image=False,
                         frame_rate=frame_rate,
                         mode='track')
            else:  # input video, do detection
                eval_seq(opt=opt,
                         data_loader=data_loader,
                         data_type='mot',
                         result_f_name=result_file_name,
                         save_dir=frame_dir,
                         show_image=False,
                         frame_rate=frame_rate,
                         mode='detect')
        else:
            # only for tmp detection evaluation...
            output_dir = '/users/duanyou/c5/results_new/results_all/tmp'
            eval_imgs_output_dets(opt=opt,
                                  data_loader=data_loader,
                                  data_type='mot',
                                  result_f_name=result_file_name,
                                  out_dir=output_dir,
                                  save_dir=frame_dir,
                                  show_image=False)
    except Exception as e:
        logger.info(e)

    if opt.output_format == 'video':
        output_video_path = 'result.mp4'
        if opt.input_mode == 'video':
            if opt.id_weight > 0:
                output_video_path = osp.join(result_root, f_name + '_track.mp4')
            else:
                output_video_path = osp.join(result_root, f_name + '_det.mp4')
        elif opt.input_mode == 'image_dir':
            output_video_path = osp.join(result_root, f_name + '_det.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}' \
            .format(osp.join(result_root, 'frame'), output_video_path)
        print(cmd_str)
        os.system(cmd_str)


def test_single(img_path, dev):
    """
    :param img_path:
    :param dev:
    :return:
    """
    if not os.path.isfile(img_path):
        print('[Err]: invalid image path.')
        return

    # Head dimensions of the net
    heads = {'hm': 5, 'reg': 2, 'wh': 2, 'id': 128}

    # Load model and put to device
    net = create_model(arch='resdcn_18', heads=heads, head_conv=256)
    model_path = '/mnt/diskb/even/MCMOT/exp/mot/default/mcmot_last_det_resdcn_18.pth'
    net = load_model(model=net, model_path=model_path)
    net = net.to(dev)
    net.eval()
    print(net)

    # Read image
    img_0 = cv2.imread(img_path)  # BGR
    assert img_0 is not None, 'Failed to load ' + img_path

    # Padded resize
    h_in, w_in = 608, 1088  # (608, 1088) (320, 640)
    img, _, _, _ = letterbox(img=img_0, height=h_in, width=w_in)

    # Preprocess image: BGR -> RGB and H×W×C -> C×H×W
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    # Convert to tensor and put to device
    blob = torch.from_numpy(img).unsqueeze(0).to(dev)

    with torch.no_grad():
        # Network output
        output = net.forward(blob)[-1]

        # Tracking output
        hm = output['hm'].sigmoid_()
        reg = output['reg']
        wh = output['wh']
        id_feature = output['id']
        id_feature = F.normalize(id_feature, dim=1)  # L2 normalization for feature vector

        # Decode output
        dets, inds, cls_inds_mask = mot_decode(hm, wh, reg, 5, False, 128)

        # Get ReID feature vector by object class
        cls_id_feats = []  # topK feature vectors of each object class
        for cls_id in range(5):  # cls_id starts from 0
            # get inds of each object class
            cls_inds = inds[:, cls_inds_mask[cls_id]]

            # gather feats for each object class
            cls_id_feature = _tranpose_and_gather_feat(id_feature, cls_inds)  # inds: 1×128
            cls_id_feature = cls_id_feature.squeeze(0)  # n × FeatDim
            if dev == 'cpu':
                cls_id_feature = cls_id_feature.numpy()
            else:
                cls_id_feature = cls_id_feature.cpu().numpy()
            cls_id_feats.append(cls_id_feature)

        # Convert back to original image coordinate system
        height_0, width_0 = img_0.shape[0], img_0.shape[1]  # H, W of original input image
        dets = map2orig(dets, h_in // 4, w_in // 4, height_0, width_0, 5)  # translate and scale

        # Parse detections of each class
        dets_dict = defaultdict(list)
        for cls_id in range(5):  # cls_id start from index 0
            cls_dets = dets[cls_id]

            # filter out low conf score dets
            remain_inds = cls_dets[:, 4] > 0.4
            cls_dets = cls_dets[remain_inds]
            # cls_id_feature = cls_id_feats[cls_id][remain_inds]  # if need re-id
            dets_dict[cls_id] = cls_dets

    # Visualize detection results
    img_draw = plot_detects(img_0, dets_dict, 5, frame_id=0, fps=30.0)
    # cv2.imshow('Detection', img_draw)
    # cv2.waitKey()
    cv2.imwrite('/mnt/diskb/even/MCMOT/results/00000.jpg', img_draw)


if __name__ == '__main__':
    opt = opts().init()
    run_demo(opt)

    # test_single(img_path='/mnt/diskb/even/MCMOT/src/00000.jpg', dev=torch.device('cpu'))
