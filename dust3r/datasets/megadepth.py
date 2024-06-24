# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed MegaDepth
# dataset at https://www.cs.cornell.edu/projects/megadepth/
# See datasets_preprocess/preprocess_megadepth.py
# --------------------------------------------------------
import os.path as osp
import numpy as np

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class MegaDepth(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data(self.split)

        if self.split is None:
            pass
        elif self.split == 'train':
            self.select_scene(('0015', '0022'), opposite=True)
        elif self.split == 'val':
            self.select_scene(('0015', '0022'))
        else:
            raise ValueError(f'bad {self.split=}')

    def _load_data(self, split):
        with np.load(osp.join(self.ROOT, 'all_metadata.npz')) as data:
            self.all_scenes = data['scenes']
            self.all_images = data['images']
            self.pairs = data['pairs']

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.all_scenes)} scenes'

    def select_scene(self, scene, *instances, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        scene_id = [s.startswith(scenes) for s in self.all_scenes]
        assert any(scene_id), 'no scene found'

        valid = np.in1d(self.pairs['scene_id'], np.nonzero(scene_id)[0])
        if instances:
            image_id = [i.startswith(instances) for i in self.all_images]
            image_id = np.nonzero(image_id)[0]
            assert len(image_id), 'no instance found'
            # both together?
            if len(instances) == 2:
                valid &= np.in1d(self.pairs['im1_id'], image_id) & np.in1d(self.pairs['im2_id'], image_id)
            else:
                valid &= np.in1d(self.pairs['im1_id'], image_id) | np.in1d(self.pairs['im2_id'], image_id)

        if opposite:
            valid = ~valid
        assert valid.any()
        self.pairs = self.pairs[valid]

    def _get_views(self, pair_idx, resolution, rng):
        scene_id, im1_id, im2_id, score = self.pairs[pair_idx]

        scene, subscene = self.all_scenes[scene_id].split()
        seq_path = osp.join(self.ROOT, scene, subscene)

        views = []

        for im_id in [im1_id, im2_id]:
            img = self.all_images[im_id]
            try:
                image = imread_cv2(osp.join(seq_path, img + '.jpg'))
                depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
                camera_params = np.load(osp.join(seq_path, img + ".npz"))
            except Exception as e:
                raise OSError(f'cannot load {img}, got exception {e}')

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.float32(camera_params['cam2world'])

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(seq_path, img))

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset='MegaDepth',
                label=osp.relpath(seq_path, self.ROOT),
                instance=img))

        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = MegaDepth(split='train', ROOT="data/megadepth_processed", resolution=224, aug_crop=16)

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
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
