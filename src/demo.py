from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import logging
import os
import shutil

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch

my_visible_devs = '6'  # '0, 3'  # 设置可运行GPU编号
os.environ['CUDA_VISIBLE_DEVICES'] = my_visible_devs
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

import os.path as osp
from lib.opts import opts  # import opts
from lib.tracking_utils.utils import mkdir_if_missing
from lib.tracking_utils.log import logger
import lib.datasets.dataset.jde as datasets
from track import eval_seq, eval_seq_and_output_dets

logger.setLevel(logging.INFO)


def run_demo(opt):
    """
    :param opt:
    :return:
    """
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    # # clear existing frame results
    # frame_res_dir = result_root + '/frame'
    # if os.path.isdir(frame_res_dir):
    #     shutil.rmtree(frame_res_dir)
    #     os.makedirs(frame_res_dir)
    # else:
    #     os.makedirs(frame_res_dir)

    if opt.input_mode == 'video':
        logger.info('Starting tracking...')
        data_loader = datasets.LoadVideo(opt.input_video, opt.img_size)  # load video as input
    elif opt.input_mode == 'image_dir':
        logger.info('Starting detection...')
        data_loader = datasets.LoadImages(opt.input_img, opt.img_size)  # load images as input
    elif opt.input_mode == 'img_path_list_txt':
        if not os.path.isfile(opt.input_img):
            print('[Err]: invalid image file path list.')
            return

        with open(opt.input_img, 'r', encoding='utf-8') as r_h:
            logger.info('Starting detection...')
            paths = [x.strip() for x in r_h.readlines()]
            print('Total {:d} image files.'.format(len(paths)))
            data_loader = datasets.LoadImages(path=paths, img_size=opt.img_size)

    result_file_name = os.path.join(result_root, 'results.txt')
    frame_rate = data_loader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')

    opt.device = device
    try:  # 视频推断的入口函数
        if opt.id_weight > 0:
            eval_seq(opt=opt,
                     data_loader=data_loader,
                     data_type='mot',
                     result_f_name=result_file_name,
                     save_dir=frame_dir,
                     show_image=False,
                     frame_rate=frame_rate,
                     mode='track')
        else:
            # eval_seq(opt=opt,
            #          data_loader=data_loader,
            #          data_type='mot',
            #          result_f_name=result_file_name,
            #          save_dir=frame_dir,
            #          show_image=False,
            #          frame_rate=frame_rate,
            #          mode='detect')

            # only for tmp detection evaluation...
            output_dir = '/users/duanyou/c5/results_new/results_all/tmp'
            eval_seq_and_output_dets(opt=opt,
                                     data_loader=data_loader,
                                     data_type='mot',
                                     result_f_name=result_file_name,
                                     out_dir=output_dir,
                                     save_dir=frame_dir,
                                     show_image=False)
    except Exception as e:
        logger.info(e)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}' \
            .format(osp.join(result_root, 'frame'),
                    output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    opt = opts().init()
    run_demo(opt)
