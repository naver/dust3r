# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed StaticThings3D
# dataset at https://github.com/lmb-freiburg/robustmvd/
# See datasets_preprocess/preprocess_staticthings3d.py
# --------------------------------------------------------
import os.path as osp
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class StaticThings3D (BaseStereoViewDataset):
    """ Dataset of indoor scenes, 5 images each time
    """
    def __init__(self, ROOT, *args, mask_bg='rand', **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)

        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg

        # loading all pairs
        assert self.split is None
        self.pairs = np.load(osp.join(ROOT, 'staticthings_pairs.npy'))

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs'

    def _get_views(self, pair_idx, resolution, rng):
        scene, seq, cam1, im1, cam2, im2 = self.pairs[pair_idx]
        seq_path = osp.join('TRAIN', scene.decode('ascii'), f'{seq:04d}')

        views = []

        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        CAM = {b'l':'left', b'r':'right'}
        for cam, idx in [(CAM[cam1], im1), (CAM[cam2], im2)]:
            num = f"{idx:04n}"
            img = num+"_clean.jpg" if rng.choice(2) else num+"_final.jpg"
            image = imread_cv2(osp.join(self.ROOT, seq_path, cam, img))
            depthmap = imread_cv2(osp.join(self.ROOT, seq_path, cam, num+".exr"))
            camera_params = np.load(osp.join(self.ROOT, seq_path, cam, num+".npz"))

            intrinsics = camera_params['intrinsics']
            camera_pose = camera_params['cam2world']

            if mask_bg:
                depthmap[depthmap > 200] = 0

            image, depthmap, intrinsics = self._crop_resize_if_necessary(image, depthmap, intrinsics, resolution, rng, info=(seq_path,cam,img))

            views.append(dict(
                img = image, 
                depthmap = depthmap,
                camera_pose = camera_pose, # cam2world
                camera_intrinsics = intrinsics,
                dataset = 'StaticThings3D',
                label = seq_path,
                instance = cam+'_'+img))

        return views


if __name__ == '__main__':
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = StaticThings3D(ROOT="data/staticthings3d_processed", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(idx, view_name(views[0]), view_name(views[1]))
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
                           color=(idx*255, (1 - idx)*255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
