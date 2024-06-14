#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Preprocessing code for the StaticThings3D dataset
# dataset at https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/README.md#staticthings3d
# 1) Download StaticThings3D in /path/to/StaticThings3D/
#    with the script at https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/scripts/download_staticthings3d.sh
#    --> depths.tar.bz2 frames_finalpass.tar.bz2 poses.tar.bz2 frames_cleanpass.tar.bz2 intrinsics.tar.bz2
# 2) unzip everything in the same /path/to/StaticThings3D/ directory
# 5) python datasets_preprocess/preprocess_staticthings3d.py --StaticThings3D_dir /path/to/tmp/StaticThings3D/
# --------------------------------------------------------
import os
import os.path as osp
import re
from tqdm import tqdm
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import path_to_root  # noqa
from dust3r.utils.parallel import parallel_threads
from dust3r.datasets.utils import cropping  # noqa


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--StaticThings3D_dir', required=True)
    parser.add_argument('--precomputed_pairs', required=True)
    parser.add_argument('--output_dir', default='data/staticthings3d_processed')
    return parser


def main(db_root, pairs_path, output_dir):
    all_scenes = _list_all_scenes(db_root)

    # crop images
    args = [(db_root, osp.join(split, subsplit, seq), camera, f'{n:04d}', output_dir)
            for split, subsplit, seq in all_scenes for camera in ['left', 'right'] for n in range(6, 16)]
    parallel_threads(load_crop_and_save, args, star_args=True, front_num=1)

    # verify that all images are there
    CAM = {b'l': 'left', b'r': 'right'}
    pairs = np.load(pairs_path)
    for scene, seq, cam1, im1, cam2, im2 in tqdm(pairs):
        seq_path = osp.join('TRAIN', scene.decode('ascii'), f'{seq:04d}')
        for cam, idx in [(CAM[cam1], im1), (CAM[cam2], im2)]:
            for ext in ['clean', 'final']:
                impath = osp.join(output_dir, seq_path, cam, f"{idx:04n}_{ext}.jpg")
                assert osp.isfile(impath), f'missing an image at {impath=}'

    print(f'>> Saved all data to {output_dir}!')


def load_crop_and_save(db_root, relpath_, camera, num, out_dir):
    relpath = osp.join(relpath_, camera, num)
    if osp.isfile(osp.join(out_dir, relpath + '.npz')):
        return
    os.makedirs(osp.join(out_dir, relpath_, camera), exist_ok=True)

    # load everything
    intrinsics_in = readFloat(osp.join(db_root, 'intrinsics', relpath_, num + '.float3'))
    cam2world = np.linalg.inv(readFloat(osp.join(db_root, 'poses', relpath + '.float3')))
    depthmap_in = readFloat(osp.join(db_root, 'depths', relpath + '.float3'))
    img_clean = cv2.cvtColor(cv2.imread(osp.join(db_root, 'frames_cleanpass',
                             relpath + '.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_final = cv2.cvtColor(cv2.imread(osp.join(db_root, 'frames_finalpass',
                             relpath + '.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # do the crop
    assert img_clean.shape[:2] == (540, 960)
    assert img_final.shape[:2] == (540, 960)
    (clean_out, final_out), depthmap, intrinsics_out, R_in2out = _crop_image(
        intrinsics_in, (img_clean, img_final), depthmap_in, (512, 384))

    # write everything
    clean_out.save(osp.join(out_dir, relpath + '_clean.jpg'), quality=80)
    final_out.save(osp.join(out_dir, relpath + '_final.jpg'), quality=80)
    cv2.imwrite(osp.join(out_dir, relpath + '.exr'), depthmap)

    # New camera parameters
    cam2world[:3, :3] = cam2world[:3, :3] @ R_in2out.T
    np.savez(osp.join(out_dir, relpath + '.npz'), intrinsics=intrinsics_out, cam2world=cam2world)


def _crop_image(intrinsics_in, color_image_in, depthmap_in, resolution_out=(512, 512)):
    image, depthmap, intrinsics_out = cropping.rescale_image_depthmap(
        color_image_in, depthmap_in, intrinsics_in, resolution_out)
    R_in2out = np.eye(3)
    return image, depthmap, intrinsics_out, R_in2out


def _list_all_scenes(path):
    print('>> Listing all scenes')

    res = []
    for split in ['TRAIN']:
        for subsplit in 'ABC':
            for seq in os.listdir(osp.join(path, 'intrinsics', split, subsplit)):
                res.append((split, subsplit, seq))
    print(f'   (found ({len(res)}) scenes)')
    assert res, f'Did not find anything at {path=}'
    return res


def readFloat(name):
    with open(name, 'rb') as f:
        if (f.readline().decode("utf-8")) != 'float\n':
            raise Exception('float file %s did not contain <float> keyword' % name)

        dim = int(f.readline())

        dims = []
        count = 1
        for i in range(0, dim):
            d = int(f.readline())
            dims.append(d)
            count *= d

        dims = list(reversed(dims))
        data = np.fromfile(f, np.float32, count).reshape(dims)
    return data  # Hxw or CxHxW NxCxHxW


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.StaticThings3D_dir, args.precomputed_pairs, args.output_dir)
