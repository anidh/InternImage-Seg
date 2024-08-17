# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
import cv2
import os.path as osp
import os
import glob
import numpy as np
import cv2
import os
from tqdm import tqdm



config_path = '/media/green_7/anidh/repos/InternImage/segmentation/work_dirs/upernet_internimage_xl_512x1024_160k_iddaw/upernet_internimage_xl_512x1024_160k_iddaw.py'
checkpoint_path = '/media/green_7/anidh/repos/InternImage/segmentation/work_dirs/upernet_internimage_xl_512x1024_160k_iddaw/best_mIoU_iter_32000.pth'

try:
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_path, checkpoint=None, device='cuda:0')
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cuda:0')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)
except Exception as e:
    print(e)

indir = '/media/green_7/anidh/dataset/iddaw_test/IDDAW_test_final/RGB'
outdir = '/media/green_7/anidh/dataset/submissions/submission-idd-iddaw/submission32000'
outdir = os.path.join(outdir, 'submission')

if not os.path.exists(outdir):
    os.makedirs(outdir)

imgfiles = glob.glob(os.path.join(indir, '*.png'))

for imgfile in tqdm(imgfiles):
    res = inference_segmentor(model, imgfile)
    print(res.shape)
    res = np.squeeze(res, axis=0)
    cv2.imwrite(os.path.join(outdir, os.path.basename(imgfile).replace('rgb', 'mask')), res)