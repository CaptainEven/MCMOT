cd src
python3 train.py --task mot \
                 --exp_id all_dla34 \
                 --gpus 3 \
                 --batch_size 4 \
                 --load_model '../models/ctdet_coco_dla_2x.pth'
cd ..