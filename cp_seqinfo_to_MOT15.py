import shutil
import os

seqinfo_dir = "./seqinfo"  # seqinfo 所在的路径
MOT15_dir = "../dataset/MOT15/images/train/"  # 数据集所在的路径
seqs = ['ADL-Rundle-6',
        'ETH-Bahnhof',
        'KITTI-13',
        'PETS09-S2L1',
        'TUD-Stadtmitte',
        'ADL-Rundle-8',
        'KITTI-17',
        'ETH-Pedcross2',
        'ETH-Sunnyday',
        'TUD-Campus',
        'Venice-2']

for seq in seqs:
    src = os.path.join(seqinfo_dir, seq, "seqinfo.ini")
    dst = os.path.join(MOT15_dir, seq, "seqinfo.ini")
    shutil.copy(src, dst)

print('Done')