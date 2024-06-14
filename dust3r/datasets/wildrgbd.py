# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed WildRGB-D
# dataset at https://github.com/wildrgbd/wildrgbd/
# See datasets_preprocess/preprocess_wildrgbd.py
# --------------------------------------------------------
import os.path as osp

import cv2
import numpy as np

from dust3r.datasets.co3d import Co3d
from dust3r.utils.image import imread_cv2


class WildRGBD(Co3d):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        super().__init__(mask_bg, *args, ROOT=ROOT, **kwargs)
        self.dataset_label = 'WildRGBD'

    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'metadata', f'{view_idx:0>5d}.npz')

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'rgb', f'{view_idx:0>5d}.jpg')

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'depth', f'{view_idx:0>5d}.png')

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'masks', f'{view_idx:0>5d}.png')

    def _read_depthmap(self, depthpath, input_metadata):
        # We store depths in the depth scale of 1000.
        # That is, when we load depth image and divide by 1000, we could get depth in meters.
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / 1000.0
        return depthmap


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = WildRGBD(split='train', ROOT="data/wildrgbd_processed", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
