# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Dataloader for preprocessed ArkitScenes data in the format of Co3d_v2
# See datasets_preprocess/preprocess_arkitscenes.py

import os.path as osp
import json
import itertools
from collections import deque

import cv2
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from dust3r.datasets.utils.sampling import compute_camera_iou

class Arkit(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg

        # load all scenes
        with open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
            self.scenes = json.load(f)
        self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
        self.scenes = {(k, k2): v2 for k, v in self.scenes.items()
                        for k2, v2 in v.items()}
        self.scene_list = list(self.scenes.keys())
        print(f"{len(self.scene_list)} scenes in scene list")

        # load all pairs
        with open(osp.join(self.ROOT, f'selected_pairs_{self.split}.json'), 'r') as f:
            pairs_metadata = json.load(f)

        self.combinations = {}
        self.scene_sizes = {}
        for dataset_name,pairs_dataset in pairs_metadata.items():
            for sid,pair_info in pairs_dataset.items():
                self.combinations[(dataset_name,sid)] = pair_info
                self.scene_sizes[(dataset_name,sid)] = len(pair_info)
        self.invalidate = {scene: {} for scene in self.scene_list}

    def __len__(self):
        # return len(self.scene_list) * len(self.combinations)
        return sum([len(v) for v in self.combinations.values()])

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        max_idx = 0
        selected_scene = self.scene_list[0]
        for k in self.scene_list:
            print(k)
            # k is obj,inst pair
            new_max = max_idx + self.scene_sizes[k] - 1
            if new_max > idx:
                break
            selected_scene = k
            max_idx = new_max_idx
        within_scene_idx = idx - max_idx

        print(f"max idx: {max_idx} within scene idx: {within_scene_idx}")

        # obj, instance = self.scene_list[idx // len(self.combinations)]
        obj,instance = selected_scene
        image_pool = self.scenes[obj, instance]
        im1_idx, im2_idx,iou = self.combinations[(obj,instance)][within_scene_idx]
        print(f"selected pair with iou: {iou} ({im1_idx,im2_idx})")

        # add a bit of randomness
        last = len(image_pool)-1

        if resolution not in self.invalidate[obj, instance]:  # flag invalid images
            self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []
        imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in [im2_idx, im1_idx]]
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()

            if self.invalidate[obj, instance][resolution][im_idx]:
                # search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break

            view_idx = image_pool[im_idx]

            impath = osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.jpg')

            # load camera params
            input_metadata = np.load(impath.replace('jpg', 'npz'))
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)

            # load image and depth
            rgb_image = imread_cv2(impath)

            depthmap = imread_cv2(impath.replace('images', 'depths') + '.geometric.png', cv2.IMREAD_UNCHANGED)
            depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])

            if mask_bg:
                # load object mask
                maskpath = osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06n}.png')
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                print(maskmap.min(), maskmap.max())
                maskmap = (maskmap / 255.0) > 0.1

                # update the depthmap with mask
                depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # problem, invalidate image and retry
                self.invalidate[obj, instance][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='Co3d_v2',
                label=osp.join(obj, instance),
                instance=osp.split(impath)[1],
            ))
        return views
