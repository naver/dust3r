#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Preprocessing code for the WayMo Open dataset
# dataset at https://github.com/waymo-research/waymo-open-dataset
# 1) Accept the license
# 2) download all training/*.tfrecord files from Perception Dataset, version 1.4.2
# 3) put all .tfrecord files in '/path/to/waymo_dir'
# 4) install the waymo_open_dataset package with
#    `python3 -m pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4`
# 5) execute this script as `python preprocess_waymo.py --waymo_dir /path/to/waymo_dir`
# --------------------------------------------------------
import sys
import os
import os.path as osp
import shutil
import json
from tqdm import tqdm
import PIL.Image
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

import path_to_root  # noqa
from dust3r.utils.geometry import geotrf, inv
from dust3r.utils.image import imread_cv2
from dust3r.utils.parallel import parallel_processes as parallel_map
from dust3r.datasets.utils import cropping
from dust3r.viz import show_raw_pointcloud


def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--waymo_dir', required=True)
    parser.add_argument('--precomputed_pairs', required=True)
    parser.add_argument('--output_dir', default='data/waymo_processed')
    parser.add_argument('--workers', type=int, default=1)
    return parser


def main(waymo_root, pairs_path, output_dir, workers=1):
    extract_frames(waymo_root, output_dir, workers=workers)
    make_crops(output_dir, workers=args.workers)

    # make sure all pairs are there
    with np.load(pairs_path) as data:
        scenes = data['scenes']
        frames = data['frames']
        pairs = data['pairs']  # (array of (scene_id, img1_id, img2_id)

    for scene_id, im1_id, im2_id in pairs:
        for im_id in (im1_id, im2_id):
            path = osp.join(output_dir, scenes[scene_id], frames[im_id] + '.jpg')
            assert osp.isfile(path), f'Missing a file at {path=}\nDid you download all .tfrecord files?'

    shutil.rmtree(osp.join(output_dir, 'tmp'))
    print('Done! all data generated at', output_dir)


def _list_sequences(db_root):
    print('>> Looking for sequences in', db_root)
    res = sorted(f for f in os.listdir(db_root) if f.endswith('.tfrecord'))
    print(f'    found {len(res)} sequences')
    return res


def extract_frames(db_root, output_dir, workers=8):
    sequences = _list_sequences(db_root)
    output_dir = osp.join(output_dir, 'tmp')
    print('>> outputing result to', output_dir)
    args = [(db_root, output_dir, seq) for seq in sequences]
    parallel_map(process_one_seq, args, star_args=True, workers=workers)


def process_one_seq(db_root, output_dir, seq):
    out_dir = osp.join(output_dir, seq)
    os.makedirs(out_dir, exist_ok=True)
    calib_path = osp.join(out_dir, 'calib.json')
    if osp.isfile(calib_path):
        return

    try:
        with tf.device('/CPU:0'):
            calib, frames = extract_frames_one_seq(osp.join(db_root, seq))
    except RuntimeError:
        print(f'/!\\ Error with sequence {seq} /!\\', file=sys.stderr)
        return  # nothing is saved

    for f, (frame_name, views) in enumerate(tqdm(frames, leave=False)):
        for cam_idx, view in views.items():
            img = PIL.Image.fromarray(view.pop('img'))
            img.save(osp.join(out_dir, f'{f:05d}_{cam_idx}.jpg'))
            np.savez(osp.join(out_dir, f'{f:05d}_{cam_idx}.npz'), **view)

    with open(calib_path, 'w') as f:
        json.dump(calib, f)


def extract_frames_one_seq(filename):
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils

    print('>> Opening', filename)
    dataset = tf.data.TFRecordDataset(filename, compression_type='')

    calib = None
    frames = []

    for data in tqdm(dataset, leave=False):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        content = frame_utils.parse_range_image_and_camera_projection(frame)
        range_images, camera_projections, _, range_image_top_pose = content

        views = {}
        frames.append((frame.context.name, views))

        # once in a sequence, read camera calibration info
        if calib is None:
            calib = []
            for cam in frame.context.camera_calibrations:
                calib.append((cam.name,
                              dict(width=cam.width,
                                   height=cam.height,
                                   intrinsics=list(cam.intrinsic),
                                   extrinsics=list(cam.extrinsic.transform))))

        # convert LIDAR to pointcloud
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose)

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        cp_points_all = np.concatenate(cp_points, axis=0)

        # The distance between lidar points and vehicle frame origin.
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        for i, image in enumerate(frame.images):
            # select relevant 3D points for this view
            mask = tf.equal(cp_points_all_tensor[..., 0], image.name)
            cp_points_msk_tensor = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)

            pose = np.asarray(image.pose.transform).reshape(4, 4)
            timestamp = image.pose_timestamp

            rgb = tf.image.decode_jpeg(image.image).numpy()

            pix = cp_points_msk_tensor[..., 1:3].numpy().round().astype(np.int16)
            pts3d = points_all[mask.numpy()]

            views[image.name] = dict(img=rgb, pose=pose, pixels=pix, pts3d=pts3d, timestamp=timestamp)

        if not 'show full point cloud':
            show_raw_pointcloud([v['pts3d'] for v in views.values()], [v['img'] for v in views.values()])

    return calib, frames


def make_crops(output_dir, workers=16, **kw):
    tmp_dir = osp.join(output_dir, 'tmp')
    sequences = _list_sequences(tmp_dir)
    args = [(tmp_dir, output_dir, seq) for seq in sequences]
    parallel_map(crop_one_seq, args, star_args=True, workers=workers, front_num=0)


def crop_one_seq(input_dir, output_dir, seq, resolution=512):
    seq_dir = osp.join(input_dir, seq)
    out_dir = osp.join(output_dir, seq)
    if osp.isfile(osp.join(out_dir, '00100_1.jpg')):
        return
    os.makedirs(out_dir, exist_ok=True)

    # load calibration file
    try:
        with open(osp.join(seq_dir, 'calib.json')) as f:
            calib = json.load(f)
    except IOError:
        print(f'/!\\ Error: Missing calib.json in sequence {seq} /!\\', file=sys.stderr)
        return

    axes_transformation = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]])

    cam_K = {}
    cam_distortion = {}
    cam_res = {}
    cam_to_car = {}
    for cam_idx, cam_info in calib:
        cam_idx = str(cam_idx)
        cam_res[cam_idx] = (W, H) = (cam_info['width'], cam_info['height'])
        f1, f2, cx, cy, k1, k2, p1, p2, k3 = cam_info['intrinsics']
        cam_K[cam_idx] = np.asarray([(f1, 0, cx), (0, f2, cy), (0, 0, 1)])
        cam_distortion[cam_idx] = np.asarray([k1, k2, p1, p2, k3])
        cam_to_car[cam_idx] = np.asarray(cam_info['extrinsics']).reshape(4, 4)  # cam-to-vehicle

    frames = sorted(f[:-3] for f in os.listdir(seq_dir) if f.endswith('.jpg'))

    # from dust3r.viz import SceneViz
    # viz = SceneViz()

    for frame in tqdm(frames, leave=False):
        cam_idx = frame[-2]  # cam index
        assert cam_idx in '12345', f'bad {cam_idx=} in {frame=}'
        data = np.load(osp.join(seq_dir, frame + 'npz'))
        car_to_world = data['pose']
        W, H = cam_res[cam_idx]

        # load depthmap
        pos2d = data['pixels'].round().astype(np.uint16)
        x, y = pos2d.T
        pts3d = data['pts3d']  # already in the car frame
        pts3d = geotrf(axes_transformation @ inv(cam_to_car[cam_idx]), pts3d)
        # X=LEFT_RIGHT y=ALTITUDE z=DEPTH

        # load image
        image = imread_cv2(osp.join(seq_dir, frame + 'jpg'))

        # downscale image
        output_resolution = (resolution, 1) if W > H else (1, resolution)
        image, _, intrinsics2 = cropping.rescale_image_depthmap(image, None, cam_K[cam_idx], output_resolution)
        image.save(osp.join(out_dir, frame + 'jpg'), quality=80)

        # save as an EXR file? yes it's smaller (and easier to load)
        W, H = image.size
        depthmap = np.zeros((H, W), dtype=np.float32)
        pos2d = geotrf(intrinsics2 @ inv(cam_K[cam_idx]), pos2d).round().astype(np.int16)
        x, y = pos2d.T
        depthmap[y.clip(min=0, max=H - 1), x.clip(min=0, max=W - 1)] = pts3d[:, 2]
        cv2.imwrite(osp.join(out_dir, frame + 'exr'), depthmap)

        # save camera parametes
        cam2world = car_to_world @ cam_to_car[cam_idx] @ inv(axes_transformation)
        np.savez(osp.join(out_dir, frame + 'npz'), intrinsics=intrinsics2,
                 cam2world=cam2world, distortion=cam_distortion[cam_idx])

        # viz.add_rgbd(np.asarray(image), depthmap, intrinsics2, cam2world)
    # viz.show()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.waymo_dir, args.precomputed_pairs, args.output_dir, workers=args.workers)
