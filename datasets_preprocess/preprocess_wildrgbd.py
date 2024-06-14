#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the WildRGB-D dataset.
# Usage:
# python3 datasets_preprocess/preprocess_wildrgbd.py --wildrgbd_dir /path/to/wildrgbd
# --------------------------------------------------------

import argparse
import random
import json
import os
import os.path as osp

import PIL.Image
import numpy as np
import cv2

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import path_to_root  # noqa
import dust3r.datasets.utils.cropping as cropping  # noqa
from dust3r.utils.image import imread_cv2


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/wildrgbd_processed")
    parser.add_argument("--wildrgbd_dir", type=str, required=True)
    parser.add_argument("--train_num_sequences_per_object", type=int, default=50)
    parser.add_argument("--test_num_sequences_per_object", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--img_size", type=int, default=512,
                        help=("lower dimension will be >= img_size * 3/4, and max dimension will be >= img_size"))
    return parser


def get_set_list(category_dir, split):
    listfiles = ["camera_eval_list.json", "nvs_list.json"]

    sequences_all = {s: {k: set() for k in listfiles} for s in ['train', 'val']}
    for listfile in listfiles:
        with open(osp.join(category_dir, listfile)) as f:
            subset_lists_data = json.load(f)
            for s in ['train', 'val']:
                sequences_all[s][listfile].update(subset_lists_data[s])
    train_intersection = set.intersection(*list(sequences_all['train'].values()))
    if split == "train":
        return train_intersection
    else:
        all_seqs = set.union(*list(sequences_all['train'].values()), *list(sequences_all['val'].values()))
        return all_seqs.difference(train_intersection)


def prepare_sequences(category, wildrgbd_dir, output_dir, img_size, split, max_num_sequences_per_object,
                      output_num_frames, seed):
    random.seed(seed)
    category_dir = osp.join(wildrgbd_dir, category)
    category_output_dir = osp.join(output_dir, category)
    sequences_all = get_set_list(category_dir, split)
    sequences_all = sorted(sequences_all)

    sequences_all_tmp = []
    for seq_name in sequences_all:
        scene_dir = osp.join(wildrgbd_dir, category_dir, seq_name)
        if not os.path.isdir(scene_dir):
            print(f'{scene_dir} does not exist, skipped')
            continue
        sequences_all_tmp.append(seq_name)
    sequences_all = sequences_all_tmp
    if len(sequences_all) <= max_num_sequences_per_object:
        selected_sequences = sequences_all
    else:
        selected_sequences = random.sample(sequences_all, max_num_sequences_per_object)

    selected_sequences_numbers_dict = {}
    for seq_name in tqdm(selected_sequences, leave=False):
        scene_dir = osp.join(category_dir, seq_name)
        scene_output_dir = osp.join(category_output_dir, seq_name)
        with open(osp.join(scene_dir, 'metadata'), 'r') as f:
            metadata = json.load(f)

        K = np.array(metadata["K"]).reshape(3, 3).T
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        w, h = metadata["w"], metadata["h"]

        camera_intrinsics = np.array(
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]]
        )
        camera_to_world_path = os.path.join(scene_dir, 'cam_poses.txt')
        camera_to_world_content = np.genfromtxt(camera_to_world_path)
        camera_to_world = camera_to_world_content[:, 1:].reshape(-1, 4, 4)

        frame_idx = camera_to_world_content[:, 0]
        num_frames = frame_idx.shape[0]
        assert num_frames >= output_num_frames
        assert np.all(frame_idx == np.arange(num_frames))

        # selected_sequences_numbers_dict[seq_name] = num_frames

        selected_frames = np.round(np.linspace(0, num_frames - 1, output_num_frames)).astype(int).tolist()
        selected_sequences_numbers_dict[seq_name] = selected_frames

        for frame_id in tqdm(selected_frames):
            depth_path = os.path.join(scene_dir, 'depth', f'{frame_id:0>5d}.png')
            masks_path = os.path.join(scene_dir, 'masks', f'{frame_id:0>5d}.png')
            rgb_path = os.path.join(scene_dir, 'rgb', f'{frame_id:0>5d}.png')

            input_rgb_image = PIL.Image.open(rgb_path).convert('RGB')
            input_mask = plt.imread(masks_path)
            input_depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64)
            depth_mask = np.stack((input_depthmap, input_mask), axis=-1)
            H, W = input_depthmap.shape

            min_margin_x = min(cx, W - cx)
            min_margin_y = min(cy, H - cy)

            # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
            l, t = int(cx - min_margin_x), int(cy - min_margin_y)
            r, b = int(cx + min_margin_x), int(cy + min_margin_y)
            crop_bbox = (l, t, r, b)
            input_rgb_image, depth_mask, input_camera_intrinsics = cropping.crop_image_depthmap(
                input_rgb_image, depth_mask, camera_intrinsics, crop_bbox)

            # try to set the lower dimension to img_size * 3/4 -> img_size=512 => 384
            scale_final = ((img_size * 3 // 4) / min(H, W)) + 1e-8
            output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)
            if max(output_resolution) < img_size:
                # let's put the max dimension to img_size
                scale_final = (img_size / max(H, W)) + 1e-8
                output_resolution = np.floor(np.array([W, H]) * scale_final).astype(int)

            input_rgb_image, depth_mask, input_camera_intrinsics = cropping.rescale_image_depthmap(
                input_rgb_image, depth_mask, input_camera_intrinsics, output_resolution)
            input_depthmap = depth_mask[:, :, 0]
            input_mask = depth_mask[:, :, 1]

            camera_pose = camera_to_world[frame_id]

            # save crop images and depth, metadata
            save_img_path = os.path.join(scene_output_dir, 'rgb', f'{frame_id:0>5d}.jpg')
            save_depth_path = os.path.join(scene_output_dir, 'depth', f'{frame_id:0>5d}.png')
            save_mask_path = os.path.join(scene_output_dir, 'masks', f'{frame_id:0>5d}.png')
            os.makedirs(os.path.split(save_img_path)[0], exist_ok=True)
            os.makedirs(os.path.split(save_depth_path)[0], exist_ok=True)
            os.makedirs(os.path.split(save_mask_path)[0], exist_ok=True)

            input_rgb_image.save(save_img_path)
            cv2.imwrite(save_depth_path, input_depthmap.astype(np.uint16))
            cv2.imwrite(save_mask_path, (input_mask * 255).astype(np.uint8))

            save_meta_path = os.path.join(scene_output_dir, 'metadata', f'{frame_id:0>5d}.npz')
            os.makedirs(os.path.split(save_meta_path)[0], exist_ok=True)
            np.savez(save_meta_path, camera_intrinsics=input_camera_intrinsics,
                     camera_pose=camera_pose)

    return selected_sequences_numbers_dict


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    assert args.wildrgbd_dir != args.output_dir

    categories = sorted([
        dirname for dirname in os.listdir(args.wildrgbd_dir)
        if os.path.isdir(os.path.join(args.wildrgbd_dir, dirname, 'scenes'))
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    splits_num_sequences_per_object = [args.train_num_sequences_per_object, args.test_num_sequences_per_object]
    for split, num_sequences_per_object in zip(['train', 'test'], splits_num_sequences_per_object):
        selected_sequences_path = os.path.join(args.output_dir, f'selected_seqs_{split}.json')
        if os.path.isfile(selected_sequences_path):
            continue
        all_selected_sequences = {}
        for category in categories:
            category_output_dir = osp.join(args.output_dir, category)
            os.makedirs(category_output_dir, exist_ok=True)
            category_selected_sequences_path = os.path.join(category_output_dir, f'selected_seqs_{split}.json')
            if os.path.isfile(category_selected_sequences_path):
                with open(category_selected_sequences_path, 'r') as fid:
                    category_selected_sequences = json.load(fid)
            else:
                print(f"Processing {split} - category = {category}")
                category_selected_sequences = prepare_sequences(
                    category=category,
                    wildrgbd_dir=args.wildrgbd_dir,
                    output_dir=args.output_dir,
                    img_size=args.img_size,
                    split=split,
                    max_num_sequences_per_object=num_sequences_per_object,
                    output_num_frames=args.num_frames,
                    seed=args.seed + int("category".encode('ascii').hex(), 16),
                )
                with open(category_selected_sequences_path, 'w') as file:
                    json.dump(category_selected_sequences, file)

            all_selected_sequences[category] = category_selected_sequences
        with open(selected_sequences_path, 'w') as file:
            json.dump(all_selected_sequences, file)
